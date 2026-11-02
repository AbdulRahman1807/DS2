"""Inference and Prediction Handoff Interface for Milestone 3.2.

Loads a trained LSTM checkpoint, executes inference over sequence DataLoaders,
and streamingly writes predictions to Parquet without accumulating million-row
DataFrames in resident RAM.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import pyarrow as pa
import pyarrow.parquet as pq

# Add workspace to path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.lstm import LSTMForecaster
from models.lstm_variants import PortEmbeddedLSTMForecaster, ResidualLSTMForecaster
from models.transformer_forecaster import ResidualTransformerForecaster
from models.attention_lstm import TemporalAttentionLSTMForecaster
from data.data_module import create_dataloaders, PortSequenceDataModule


def run_inference_and_export(
    checkpoint_path: str,
    config_path: str,
    output_parquet_path: str,
    split: str = "val",
    batch_size: int = 512,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute streaming inference on a specified split and export to Parquet.
    
    Args:
        checkpoint_path: Path to best_model.pt.
        config_path: Path to sequence_pipeline.yaml.
        output_parquet_path: Destination path for prediction Parquet file.
        split: Split partition ('val', 'test', or 'train').
        batch_size: Batch size for inference evaluation.
        device: Device string ('mps', 'cpu').
        
    Returns:
        Summary metrics dictionary.
    """
    if device is None:
        device_obj = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
    else:
        device_obj = torch.device(device)

    print(f"Loading model checkpoint from {checkpoint_path} onto {device_obj}...")
    ckpt = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    m_cfg = ckpt.get("model_config", {})
    model_type = m_cfg.get("model_type", "LSTMForecaster")
    return_port_idx = (model_type == "PortEmbeddedLSTMForecaster")

    if model_type == "PortEmbeddedLSTMForecaster":
        model = PortEmbeddedLSTMForecaster(
            num_ports=m_cfg.get("num_ports", 2065),
            embedding_dim=m_cfg.get("embedding_dim", 16),
            input_size=m_cfg.get("input_size", 20),
            hidden_size=m_cfg.get("hidden_size", 64),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=m_cfg.get("dropout", 0.2),
            output_size=m_cfg.get("output_size", 1),
            output_activation=m_cfg.get("output_activation", "softplus"),
        ).to(device_obj)
    elif model_type == "ResidualLSTMForecaster":
        model = ResidualLSTMForecaster(
            input_size=m_cfg.get("input_size", 20),
            hidden_size=m_cfg.get("hidden_size", 64),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=m_cfg.get("dropout", 0.2),
            roll7_idx=m_cfg.get("roll7_idx", 12),
            roll7_mean=m_cfg.get("roll7_mean", 2.4068180554238894),
            roll7_std=m_cfg.get("roll7_std", 6.3483944613159915),
            constraint=m_cfg.get("constraint", "relu"),
        ).to(device_obj)
    elif model_type == "ResidualTransformerForecaster":
        model = ResidualTransformerForecaster(
            input_size=m_cfg.get("input_size", 20),
            seq_len=m_cfg.get("seq_len", 30),
            d_model=m_cfg.get("d_model", 64),
            nhead=m_cfg.get("nhead", 4),
            num_layers=m_cfg.get("num_layers", 2),
            dim_feedforward=m_cfg.get("dim_feedforward", 128),
            dropout=m_cfg.get("dropout", 0.1),
            pooling=m_cfg.get("pooling", "last"),
            pe_type=m_cfg.get("pe_type", "sinusoidal"),
            roll7_idx=m_cfg.get("roll7_idx", 12),
            roll7_mean=m_cfg.get("roll7_mean", 2.4068180554238894),
            roll7_std=m_cfg.get("roll7_std", 6.3483944613159915),
            constraint=m_cfg.get("constraint", "relu"),
        ).to(device_obj)
    elif model_type == "TemporalAttentionLSTMForecaster":
        model = TemporalAttentionLSTMForecaster(
            input_size=m_cfg.get("input_size", 20),
            hidden_size=m_cfg.get("hidden_size", 64),
            attention_dim=m_cfg.get("attention_dim", 32),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=m_cfg.get("dropout", 0.2),
            roll7_idx=m_cfg.get("roll7_idx", 12),
            roll7_mean=m_cfg.get("roll7_mean", 2.4068180554238894),
            roll7_std=m_cfg.get("roll7_std", 6.3483944613159915),
            constraint=m_cfg.get("constraint", "relu"),
        ).to(device_obj)
    else:
        model = LSTMForecaster(
            input_size=m_cfg.get("input_size", 20),
            hidden_size=m_cfg.get("hidden_size", 64),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=m_cfg.get("dropout", 0.2),
            output_size=m_cfg.get("output_size", 1),
            output_activation=m_cfg.get("output_activation", "softplus"),
        ).to(device_obj)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Setting up DataLoader for split='{split}' with batch_size={batch_size}...")
    loaders = create_dataloaders(
        config_path=config_path,
        batch_size=batch_size,
        num_workers=0,
        return_port_idx=return_port_idx,
    )
    data_loader = loaders[split]
    dataset = data_loader.dataset
    df_meta = dataset.df_meta

    total_samples = len(dataset)
    print(f"Total sequences to evaluate: {total_samples:,}")

    # Prepare PyArrow writer for streaming bounded output
    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrow_schema = pa.schema([
        ("sequence_id", pa.string()),
        ("portwatch_id", pa.string()),
        ("reference_date", pa.string()),
        ("split_tag", pa.string()),
        ("y_true", pa.float32()),
        ("y_pred", pa.float32()),
    ])

    writer = pq.ParquetWriter(str(output_path), arrow_schema, compression="SNAPPY")

    t0 = time.time()
    processed_count = 0
    all_preds = []
    
    # Process batches
    chunk_seq_ids = []
    chunk_port_ids = []
    chunk_ref_dates = []
    chunk_split_tags = []
    chunk_y_true = []
    chunk_y_pred = []
    chunk_size = 50000

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if len(batch) == 3:
                bx, bp, by = batch
                bx = bx.to(device_obj)
                bp = bp.to(device_obj)
                preds = model(bx, bp).cpu().view(-1).numpy()
            else:
                bx, by = batch
                bx = bx.to(device_obj)
                preds = model(bx).cpu().view(-1).numpy()
            y_vals = by.view(-1).numpy()

            b_len = len(preds)
            idx_start = processed_count
            idx_end = processed_count + b_len
            
            sub_meta = df_meta.iloc[idx_start:idx_end]
            chunk_seq_ids.extend(sub_meta["sequence_id"].tolist())
            chunk_port_ids.extend(sub_meta["portwatch_id"].tolist())
            chunk_ref_dates.extend(sub_meta["reference_date"].tolist())
            chunk_split_tags.extend(sub_meta["split_tag"].tolist())
            chunk_y_true.extend(y_vals.tolist())
            chunk_y_pred.extend(preds.tolist())

            processed_count += b_len

            if len(chunk_seq_ids) >= chunk_size or processed_count == total_samples:
                table = pa.Table.from_arrays(
                    [
                        pa.array(chunk_seq_ids, type=pa.string()),
                        pa.array(chunk_port_ids, type=pa.string()),
                        pa.array(chunk_ref_dates, type=pa.string()),
                        pa.array(chunk_split_tags, type=pa.string()),
                        pa.array(chunk_y_true, type=pa.float32()),
                        pa.array(chunk_y_pred, type=pa.float32()),
                    ],
                    schema=arrow_schema,
                )
                writer.write_table(table)
                chunk_seq_ids.clear()
                chunk_port_ids.clear()
                chunk_ref_dates.clear()
                chunk_split_tags.clear()
                chunk_y_true.clear()
                chunk_y_pred.clear()

            if (batch_idx + 1) % 500 == 0 or processed_count == total_samples:
                rate = processed_count / (time.time() - t0)
                print(f"  Processed {processed_count:,}/{total_samples:,} sequences ({rate:,.0f} seq/s)...")

    writer.close()
    elapsed = time.time() - t0
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("=" * 80)
    print(f"Inference and export complete in {elapsed:.2f}s ({total_samples/elapsed:,.0f} seq/s)")
    print(f"Predictions saved to: {output_path} ({file_size_mb:.2f} MB)")
    print("=" * 80)

    return {
        "output_file": str(output_path),
        "total_sequences": total_samples,
        "elapsed_sec": elapsed,
        "throughput_seq_s": total_samples / elapsed,
        "file_size_mb": file_size_mb,
    }


if __name__ == "__main__":
    import yaml
    base_dir = Path(__file__).resolve().parent
    cfg_file = base_dir / "configs/lstm_baseline.yaml"
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)

    ckpt_file = base_dir / "artifacts/lstm/best_model.pt"
    pipe_cfg = base_dir / "configs/sequence_pipeline.yaml"
    out_file = base_dir / "artifacts/lstm/validation_predictions.parquet"

    run_inference_and_export(
        checkpoint_path=str(ckpt_file),
        config_path=str(pipe_cfg),
        output_parquet_path=str(out_file),
        split="val",
        batch_size=1024,
    )

