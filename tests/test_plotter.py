import pytest
import pandas as pd
import os
from src.plotter import MovingAveragePlotter

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Close': [100, 101, 102, 103, 104]
    }, index=pd.date_range('2024-01-01', periods=5))

def test_calculate_ma(sample_data):
    plotter = MovingAveragePlotter(sample_data, symbol="AAPL")
    plotter.calculate_ma([2])
    
    assert 'MA2' in plotter.data.columns
    assert pd.isna(plotter.data['MA2'].iloc[0])
    assert plotter.data['MA2'].iloc[1] == 100.5

def test_plot_success(sample_data, tmp_path):
    plotter = MovingAveragePlotter(sample_data, symbol="AAPL")
    plot_path = plotter.plot([2], str(tmp_path))
    
    assert plot_path is not None
    assert os.path.exists(plot_path)
    assert plot_path.endswith("AAPL_moving_averages.pdf")