# Cell2Sentence Development Guide

## Overview
Cell2Sentence (C2S) applies Large Language Models to single-cell transcriptomics by transforming gene expression vectors into **cell sentences**—space-separated gene names ordered by descending expression. This allows LLMs to natively model scRNA-seq data using natural language.

## Architecture & Core Components

### Three-Layer Design Pattern
The codebase follows a consistent layering:

1. **Data Layer (`csdata.py`)**: `CSData` wraps HuggingFace Arrow datasets, managing vocabulary (gene names → cell counts) and cell sentences. Always access data through CSData objects, not raw files.

2. **Model Layer (`csmodel.py`)**: `CSModel` wraps HuggingFace transformers, handling tokenization, training, inference, and embeddings. Models are always saved to disk first, then reloaded for operations.

3. **Task Layer (`tasks.py`)**: High-level functions for end-user workflows (cell type prediction, cell generation, embedding). These orchestrate CSData + CSModel interactions.

### Prompt Formatting System
Located in `src/cell2sentence/prompts/`, this is a **template engine with variable substitution**:
- JSON files define task-specific prompt variations with `{key}` placeholders
- `PromptFormatter` classes (in `prompt_formatter.py`) replace keys with actual values
- Keys include: `{num_genes}`, `{organism}`, `{cell_type}`, `{cell_sentence}`
- Multiple prompt variations per task provide linguistic diversity during training

**Critical**: Prompts are formatted at dataset preparation time, before tokenization. See `C2SPromptFormatter.format_hf_ds()`.

## Critical Workflows

### Data Preparation Pipeline (Tutorial 0)
```python
# 1. Standard preprocessing with BASE-10 log1p (not natural log!)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata, base=10)  # Base 10 is essential for reconstruction

# 2. Convert to Arrow dataset
arrow_ds, vocabulary = CSData.adata_to_arrow(adata, label_col_names=["cell_type"])

# 3. Wrap in CSData (saves to disk)
csdata = CSData.csdata_from_arrow(arrow_ds, vocabulary, save_dir, save_name)
```

**Why base-10**: Empirically optimal for inverse reconstruction (cell sentences → expression vectors). Non-negotiable.

### Model Fine-tuning Pattern
```python
# 1. Create CSModel (downloads and saves model)
csmodel = CSModel("EleutherAI/pythia-410m", save_dir, save_name)

# 2. Fine-tune (model reloaded from disk)
csmodel.fine_tune(
    csdata, 
    task="cell_type_prediction",
    train_args=TrainingArguments(...),
    loss_on_response_only=True,  # Default: compute loss only on model's answer
    top_k_genes=100
)
```

**Pattern note**: Models are saved immediately on creation, then reloaded for all operations. This ensures consistency and enables distributed training.

### Multi-Cell Tasks
Handle multiple cells as a single prompt (e.g., tissue prediction from cell population):
- Use `C2SMultiCellPromptFormatter` with `multi_cell_indices_ds` parameter
- Format: "Cell 1:\nGENE1 GENE2...\nCell 2:\nGENE3 GENE4..."
- See Tutorial 8 for tissue prediction example

## Development Conventions

### Testing with pytest
```bash
make test  # Runs pytest on src/cell2sentence/tests/
```
Tests use small CSV fixtures (`small_data.csv`, `immune_tissue_10cells.h5ad`) in `tests/` directory. Tests validate:
- Data transformation correctness (cell sentences match expected)
- Model reload from disk
- Dataset splits and vocabulary concatenation

### Build & Documentation
```bash
make install    # Editable install with pip -e
make html       # Build Sphinx docs (ReadTheDocs theme)
make lint       # Run pylint
```

Docs use Sphinx autodoc with Napoleon (Google/NumPy docstrings). See `docs/source/conf.py`.

### Flash Attention Support (Optional)
```python
# For faster generation with long sequences (>200 genes):
pip install flash-attn --no-build-isolation

# Enable in tasks:
generate_cells_conditioned_on_cell_type(..., use_flash_attn=True)
```

**Note**: Data collator in `csmodel.py` assumes NO flash attention (pads to batch max). Flash attention only used in generation, not training.

## Key Gotchas

1. **Gene Names, Not IDs**: Always use gene names (e.g., "GAPDH") in `adata.var_names`, not Ensembl IDs. Code warns if it detects "ENS" prefix.

2. **Vocabulary Ordering Matters**: `vocabulary` is an OrderedDict of {gene_name: num_cells_expressed}. Order is preserved through concatenation (`concat_vocabularies`).

3. **Reserved Words**: Avoid using `cell_name` or `cell_sentence` as metadata column names (see `RESERVED_WORDS` in `utils.py`).

4. **Label Columns**: Pass `label_col_names=["cell_type", "tissue"]` to `adata_to_arrow()` to include metadata in Arrow dataset.

5. **Context Window**: Models cap sequences at `model.config.max_position_embeddings`. Long cell sentences are truncated during tokenization.

6. **Loss Masking**: `loss_on_response_only=True` sets `labels=-100` for prompt tokens, computing loss only on model's answer. See `tokenize_loss_on_response()` in `utils.py`.

## Adding New Tasks

1. Create prompt JSON in `src/cell2sentence/prompts/`:
   ```json
   {
     "model_input": ["Template with {placeholders}"],
     "response": ["{expected_output}"]
   }
   ```

2. Add task name to `SUPPORTED_TASKS` or `MULTICELL_SUPPORTED_TASKS` in `prompt_formatter.py`

3. Implement key mapping in `PromptFormatter.get_keys_for_task()`

4. Optionally add high-level function in `tasks.py`

## Model Zoo Reference
Pretrained models on HuggingFace:
- **C2S-Pythia-410m**: Cell type prediction/generation (57M cells from CellxGene/HCA)
- **C2S-Scale-1B**: Multi-task model with paper abstract generation
- **C2S-Scale-Gemma-2B/27B**: Latest scale-up with enhanced capabilities

All models expect cell sentences as input, formatted via appropriate prompt templates.
