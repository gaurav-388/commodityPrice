"""
Model Evaluation and Visualization Module
Creates comprehensive visualizations and performance reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


class ModelEvaluator:
    def __init__(self, output_dir='results'):
        """
        Initialize evaluator
        
        Args:
            output_dir: Directory to save plots and results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_predictions(self, y_true, y_pred, title='Actual vs Predicted Prices'):
        """
        Plot actual vs predicted values
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.3, s=10)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Price (Rs)', fontsize=12)
        plt.ylabel('Predicted Price (Rs)', fontsize=12)
        plt.title('Actual vs Predicted', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Residual plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.3, s=10)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Price (Rs)', fontsize=12)
        plt.ylabel('Residuals (Rs)', fontsize=12)
        plt.title('Residual Plot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'predictions_plot.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved: {filepath}")
        plt.close()
    
    def plot_error_distribution(self, y_true, y_pred):
        """
        Plot error distribution
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        """
        errors = y_true - y_pred
        percentage_errors = (errors / y_true) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Absolute errors histogram
        axes[0, 0].hist(np.abs(errors), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Absolute Error (Rs)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Distribution of Absolute Errors', fontsize=12, fontweight='bold')
        axes[0, 0].axvline(np.mean(np.abs(errors)), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(np.abs(errors)):.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Percentage errors histogram
        axes[0, 1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Percentage Error (%)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Distribution of Percentage Errors', fontsize=12, fontweight='bold')
        axes[0, 1].axvline(np.mean(percentage_errors), color='r', linestyle='--',
                          label=f'Mean: {np.mean(percentage_errors):.2f}%')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error vs actual price
        axes[1, 0].scatter(y_true, np.abs(errors), alpha=0.3, s=10)
        axes[1, 0].set_xlabel('Actual Price (Rs)', fontsize=11)
        axes[1, 0].set_ylabel('Absolute Error (Rs)', fontsize=11)
        axes[1, 0].set_title('Error vs Actual Price', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot for normality
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'error_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved: {filepath}")
        plt.close()
    
    def plot_feature_importance(self, importance_df, top_n=20):
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to plot
        """
        importance_df = importance_df.head(top_n)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
        # Gain importance
        axes[0].barh(importance_df['feature'], importance_df['gain'], color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Gain', fontsize=11)
        axes[0].set_title('Feature Importance (Gain)', fontsize=12, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Weight importance
        axes[1].barh(importance_df['feature'], importance_df['weight'], color='lightcoral', edgecolor='black')
        axes[1].set_xlabel('Weight', fontsize=11)
        axes[1].set_title('Feature Importance (Weight)', fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Cover importance
        axes[2].barh(importance_df['feature'], importance_df['cover'], color='lightgreen', edgecolor='black')
        axes[2].set_xlabel('Cover', fontsize=11)
        axes[2].set_title('Feature Importance (Cover)', fontsize=12, fontweight='bold')
        axes[2].invert_yaxis()
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved: {filepath}")
        plt.close()
    
    def plot_price_trends(self, y_true, y_pred, dates=None, n_samples=500):
        """
        Plot actual vs predicted price trends over time
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Date values (optional)
            n_samples: Number of samples to plot
        """
        # Sample data if too large
        if len(y_true) > n_samples:
            indices = np.random.choice(len(y_true), n_samples, replace=False)
            indices = np.sort(indices)
            y_true_sample = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred
            indices = range(len(y_true))
        
        plt.figure(figsize=(14, 6))
        
        x_axis = list(range(len(y_true_sample)))
        
        plt.plot(x_axis, y_true_sample, label='Actual', alpha=0.7, linewidth=1.5, color='blue')
        plt.plot(x_axis, y_pred_sample, label='Predicted', alpha=0.7, linewidth=1.5, color='red')
        plt.fill_between(x_axis, y_true_sample, y_pred_sample, alpha=0.2, color='gray')
        
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Price (Rs)', fontsize=12)
        plt.title('Actual vs Predicted Price Trends', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'price_trends.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Price trends plot saved: {filepath}")
        plt.close()
    
    def generate_metrics_report(self, metrics, save_path='results/metrics_report.txt'):
        """
        Generate and save metrics report
        
        Args:
            metrics: Dictionary of evaluation metrics
            save_path: Path to save the report
        """
        report = []
        report.append("="*60)
        report.append("MODEL PERFORMANCE REPORT")
        report.append("="*60)
        report.append("")
        
        for metric_name, value in metrics.items():
            if 'Score' in metric_name:
                report.append(f"{metric_name:.<40} {value:.6f}")
            elif '%' in str(value) or 'MAPE' in metric_name:
                report.append(f"{metric_name:.<40} {value:.2f}%")
            else:
                report.append(f"{metric_name:.<40} {value:.2f}")
        
        report.append("")
        report.append("="*60)
        
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"\nMetrics report saved: {save_path}")
    
    def create_comprehensive_report(self, y_true, y_pred, metrics, importance_df):
        """
        Create comprehensive evaluation report with all visualizations
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            metrics: Dictionary of evaluation metrics
            importance_df: Feature importance DataFrame
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        
        # Generate all plots
        self.plot_predictions(y_true, y_pred)
        self.plot_error_distribution(y_true, y_pred)
        self.plot_feature_importance(importance_df)
        self.plot_price_trends(y_true, y_pred)
        
        # Generate metrics report
        self.generate_metrics_report(metrics)
        
        # Save feature importance to CSV
        importance_path = os.path.join(self.output_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved: {importance_path}")
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            'actual': y_true.values if hasattr(y_true, 'values') else y_true,
            'predicted': y_pred,
            'error': (y_true.values if hasattr(y_true, 'values') else y_true) - y_pred,
            'percentage_error': ((y_true.values if hasattr(y_true, 'values') else y_true) - y_pred) / 
                               (y_true.values if hasattr(y_true, 'values') else y_true) * 100
        })
        predictions_path = os.path.join(self.output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved: {predictions_path}")
        
        print("\n" + "="*60)
        print("EVALUATION REPORT COMPLETE")
        print(f"All results saved in: {self.output_dir}/")
        print("="*60)


if __name__ == "__main__":
    print("Model Evaluator Module")
    print("This module should be imported and used with trained model predictions")
