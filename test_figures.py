import pytest
import os
import pandas as pd
from collections import defaultdict
from cv_exp_all_domains import main, parse_args
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_metrics():
    """Mock metrics data that would be collected during training"""
    return [
        {'moe': False, 'add_tokens': True, 'cv_split': 0, 'f1': 0.85, 'sim_ratio': 0.75},
        {'moe': False, 'add_tokens': True, 'cv_split': 1, 'f1': 0.87, 'sim_ratio': 0.77},
        {'moe': True, 'add_tokens': True, 'cv_split': 0, 'f1': 0.90, 'sim_ratio': 0.80},
        {'moe': True, 'add_tokens': True, 'cv_split': 1, 'f1': 0.92, 'sim_ratio': 0.82},
        {'moe': True, 'add_tokens': False, 'cv_split': 0, 'f1': 0.88, 'sim_ratio': 0.78},
        {'moe': True, 'add_tokens': False, 'cv_split': 1, 'f1': 0.89, 'sim_ratio': 0.79},
    ]

def test_results_table_format(tmp_path, mock_metrics):
    """Test that results are properly formatted and saved"""
    # Group metrics by experiment setting
    grouped_results = defaultdict(list)
    for entry in mock_metrics:
        key = (entry['moe'], entry['add_tokens'])
        grouped_results[key].append(entry)
    
    # Create table rows
    table_rows = []
    for (moe_setting, add_tokens_setting), entries in grouped_results.items():
        f1_values = [e['f1'] for e in entries]
        sim_values = [e['sim_ratio'] for e in entries]
        f1_mean = sum(f1_values) / len(f1_values)
        f1_std = (sum((x - f1_mean) ** 2 for x in f1_values) / len(f1_values)) ** 0.5
        sim_mean = sum(sim_values) / len(sim_values)
        sim_std = (sum((x - sim_mean) ** 2 for x in sim_values) / len(sim_values)) ** 0.5
        
        table_rows.append({
            'MOE': str(moe_setting),
            'Add_Tokens': str(add_tokens_setting),
            'F1_mean': f"{f1_mean:.4f}",
            'F1_std': f"{f1_std:.4f}",
            'Sim_Ratio_mean': f"{sim_mean:.4f}",
            'Sim_Ratio_std': f"{sim_std:.4f}"
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(table_rows)
    results_file = os.path.join(tmp_path, "final_results.csv")
    df.to_csv(results_file, index=False)
    
    # Read and verify the saved CSV
    df_read = pd.read_csv(results_file)
    
    # Verify table format and content
    assert 'MOE' in df_read.columns
    assert 'Add_Tokens' in df_read.columns
    assert 'F1_mean' in df_read.columns
    assert 'F1_std' in df_read.columns
    assert 'Sim_Ratio_mean' in df_read.columns
    assert 'Sim_Ratio_std' in df_read.columns
    
    # Verify all experiment configurations are present
    assert 'False' in df_read['MOE'].values.astype(str)
    assert 'True' in df_read['MOE'].values.astype(str)
    

@pytest.mark.integration
def test_main_results_saving():
    """Integration test for the main function's results saving"""
    # Create mock arguments
    mock_args = MagicMock()
    mock_args.save_path = "test_output"
    mock_args.token = None
    mock_args.batch_size = 4
    mock_args.max_length = 32
    mock_args.save_every = 100
    mock_args.bugfix = True
    mock_args.fp16 = False
    mock_args.wandb_project = "test_project"
    
    # Mock the training process
    with patch('cv_exp_all_domains.prepare_model') as mock_prepare, \
         patch('cv_exp_all_domains.get_all_train_data') as mock_get_data, \
         patch('cv_exp_all_domains.Trainer') as mock_trainer, \
         patch('cv_exp_all_domains.wandb') as mock_wandb:
        
        # Configure mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_prepare.return_value = (mock_model, mock_tokenizer)
        mock_trainer.return_value.evaluate.return_value = {'f1': 0.9, 'sim_ratio': 0.8}
        mock_get_data.return_value = [MagicMock() for _ in range(5)]  # 5-fold CV
        
        # Mock trainer instance
        trainer_instance = MagicMock()
        trainer_instance.evaluate.return_value = {'f1': 0.9, 'sim_ratio': 0.8}
        trainer_instance.train.return_value = None
        mock_trainer.return_value = trainer_instance
        
        # Create test output directory
        os.makedirs(mock_args.save_path, exist_ok=True)
        
        try:
            # Run main function
            main(mock_args)
            
            # Verify results file was created
            results_file = os.path.join(mock_args.save_path, "final_results.csv")
            assert os.path.exists(results_file)
            
            # Verify content of results file
            df = pd.read_csv(results_file)
            assert 'MOE' in df.columns
            assert 'Add_Tokens' in df.columns
            assert 'F1_mean' in df.columns
            assert 'F1_std' in df.columns
            assert 'Sim_Ratio_mean' in df.columns
            assert 'Sim_Ratio_std' in df.columns
        
        finally:
            # cleanup
            if os.path.exists(mock_args.save_path):
                import shutil
                shutil.rmtree(mock_args.save_path)


if __name__ == "__main__":
    pytest.main([__file__])
