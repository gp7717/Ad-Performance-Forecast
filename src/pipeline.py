"""
End-to-end pipeline for ad performance forecasting
"""
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

from config import config
from fetch.fetch_meta_insights import fetch_meta_insights
from features.prepare_dataset import prepare_dataset
from features.feature_engineering import engineer_features
from models.train_lgbm import train_lgbm_model
from models.predict_next12h import predict_next_12h
from models.evaluate import evaluate_model


class ForecastPipeline:
    """End-to-end forecasting pipeline."""

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def run_full_pipeline(
        self,
        fetch_data: bool = True,
        prepare_features: bool = True,
        train_model: bool = True,
        make_predictions: bool = True,
        evaluate_model_flag: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the complete forecasting pipeline."""
        results = {}

        self.logger.info("Starting ad performance forecasting pipeline...")

        try:
            # 1. Fetch data
            if fetch_data:
                self.logger.info("Step 1: Fetching data...")
                # Filter kwargs for fetch method (only data-related parameters)
                fetch_kwargs = {k: v for k, v in kwargs.items()
                               if k in ['ad_account_id', 'access_token', 'since', 'until', 'output_path']}
                fetch_meta_insights(**fetch_kwargs)
                results['data_fetched'] = True

            # 2. Prepare dataset
            if prepare_features:
                self.logger.info("Step 2: Preparing features...")
                # Filter kwargs for prepare method (only data-related parameters)
                prepare_kwargs = {k: v for k, v in kwargs.items()
                                 if k in ['input_path', 'output_path']}
                df = prepare_dataset(**prepare_kwargs)
                results['features_prepared'] = True
                results['dataset_shape'] = df.shape

            # 3. Train model
            if train_model:
                self.logger.info("Step 3: Training model...")
                # Filter kwargs for train method (only training-related parameters)
                train_kwargs = {k: v for k, v in kwargs.items()
                               if k in ['data_path', 'model_path', 'n_trials']}
                model, metrics = train_lgbm_model(**train_kwargs)
                results['model_trained'] = True
                results['training_metrics'] = metrics

            # 4. Make predictions
            if make_predictions:
                self.logger.info("Step 4: Making predictions...")
                # Filter kwargs for predict method (only prediction-related parameters)
                predict_kwargs = {k: v for k, v in kwargs.items()
                                 if k in ['data_path', 'model_path', 'output_path']}
                predictions = predict_next_12h(**predict_kwargs)
                results['predictions_made'] = True
                results['predictions'] = predictions

            # 5. Evaluate model
            if evaluate_model_flag:
                self.logger.info("Step 5: Evaluating model...")
                # Filter kwargs for evaluate method (only evaluation-related parameters)
                eval_kwargs = {k: v for k, v in kwargs.items()
                              if k in ['data_path', 'model_path', 'output_dir']}
                eval_results = evaluate_model(**eval_kwargs)
                results['model_evaluated'] = True
                results['evaluation_results'] = eval_results

            self.logger.info("Pipeline completed successfully!")
            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def run_fetch_only(self, **kwargs) -> Dict[str, Any]:
        """Run only the data fetching step."""
        self.logger.info("Running data fetch pipeline...")
        fetch_kwargs = {k: v for k, v in kwargs.items()
                       if k in ['ad_account_id', 'access_token', 'since', 'until', 'output_path']}
        fetch_meta_insights(**fetch_kwargs)
        return {'data_fetched': True}

    def run_training_only(self, **kwargs) -> Dict[str, Any]:
        """Run only the model training step."""
        self.logger.info("Running model training pipeline...")
        train_kwargs = {k: v for k, v in kwargs.items()
                       if k in ['data_path', 'model_path', 'n_trials']}
        model, metrics = train_lgbm_model(**train_kwargs)
        return {'model_trained': True, 'training_metrics': metrics}

    def run_prediction_only(self, **kwargs) -> Dict[str, Any]:
        """Run only the prediction step."""
        self.logger.info("Running prediction pipeline...")
        predict_kwargs = {k: v for k, v in kwargs.items()
                         if k in ['data_path', 'model_path', 'output_path']}
        predictions = predict_next_12h(**predict_kwargs)
        return {'predictions_made': True, 'predictions': predictions}

    def run_evaluation_only(self, **kwargs) -> Dict[str, Any]:
        """Run only the evaluation step."""
        self.logger.info("Running model evaluation pipeline...")
        eval_kwargs = {k: v for k, v in kwargs.items()
                      if k in ['data_path', 'model_path', 'output_dir']}
        eval_results = evaluate_model(**eval_kwargs)
        return {'model_evaluated': True, 'evaluation_results': eval_results}


def run_pipeline(args=None):
    """Main pipeline runner function."""
    parser = argparse.ArgumentParser(description='Ad Performance Forecasting Pipeline')

    # Pipeline steps
    parser.add_argument('--fetch', action='store_true', help='Fetch new data from Meta API')
    parser.add_argument('--prepare', action='store_true', help='Prepare and clean dataset')
    parser.add_argument('--train', action='store_true', help='Train forecasting model')
    parser.add_argument('--predict', action='store_true', help='Make 12h predictions')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model performance')

    # Run all steps by default
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')

    # Configuration options
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--model-dir', type=str, help='Model directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')

    # Data parameters
    parser.add_argument('--ad-account-id', type=str, help='Meta Ads account ID')
    parser.add_argument('--access-token', type=str, help='Meta Ads access token')
    parser.add_argument('--since', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--until', type=str, help='End date (YYYY-MM-DD)')

    # Model parameters
    parser.add_argument('--n-trials', type=int, default=100, help='Number of hyperparameter trials')

    parsed_args = parser.parse_args(args)

    # Load configuration
    if parsed_args.config:
        config_path = parsed_args.config
    else:
        config_path = "config.yaml"

    # Override config with command line arguments
    if parsed_args.data_dir:
        config.set('data.raw_dir', parsed_args.data_dir + '/raw')
        config.set('data.processed_dir', parsed_args.data_dir + '/processed')
        config.set('data.models_dir', parsed_args.data_dir + '/models')
        config.set('data.predictions_dir', parsed_args.data_dir + '/predictions')

    if parsed_args.model_dir:
        config.set('data.models_dir', parsed_args.model_dir)

    if parsed_args.output_dir:
        config.set('data.predictions_dir', parsed_args.output_dir)

    # Prepare pipeline arguments
    pipeline_args = {}

    # Data parameters
    if parsed_args.ad_account_id:
        pipeline_args['ad_account_id'] = parsed_args.ad_account_id
    if parsed_args.access_token:
        pipeline_args['access_token'] = parsed_args.access_token
    if parsed_args.since:
        pipeline_args['since'] = parsed_args.since
    if parsed_args.until:
        pipeline_args['until'] = parsed_args.until

    # Path parameters
    if parsed_args.data_dir:
        data_dir = parsed_args.data_dir
        pipeline_args['input_path'] = f"{data_dir}/raw/meta_insights_hourly_dataset.csv"
        pipeline_args['data_path'] = f"{data_dir}/processed/processed_dataset.csv"
        pipeline_args['output_path'] = f"{data_dir}/predictions/spend_forecast_12h.csv"

    if parsed_args.model_dir:
        pipeline_args['model_path'] = f"{parsed_args.model_dir}/lgbm_model.pkl"

    if parsed_args.output_dir:
        pipeline_args['output_path'] = f"{parsed_args.output_dir}/spend_forecast_12h.csv"
        pipeline_args['output_dir'] = parsed_args.output_dir

    # Model parameters
    if parsed_args.n_trials:
        pipeline_args['n_trials'] = parsed_args.n_trials

    # Initialize pipeline
    pipeline = ForecastPipeline()

    # Determine which steps to run
    if parsed_args.all or (not any([parsed_args.fetch, parsed_args.prepare,
                                   parsed_args.train, parsed_args.predict, parsed_args.evaluate])):
        # Run full pipeline
        results = pipeline.run_full_pipeline(**pipeline_args)

    else:
        results = {}

        if parsed_args.fetch:
            results.update(pipeline.run_fetch_only(**pipeline_args))

        if parsed_args.prepare:
            # For prepare step, we need to fetch first or assume data exists
            if not results.get('data_fetched', False):
                pipeline.logger.warning("Prepare step requires data. Consider running --fetch first.")
            else:
                prepare_kwargs = {k: v for k, v in pipeline_args.items()
                                 if k in ['input_path', 'output_path']}
                df = prepare_dataset(**prepare_kwargs)
                results['features_prepared'] = True

        if parsed_args.train:
            train_kwargs = {k: v for k, v in pipeline_args.items()
                           if k in ['data_path', 'model_path', 'n_trials']}
            model, metrics = train_lgbm_model(**train_kwargs)
            results.update({'model_trained': True, 'training_metrics': metrics})

        if parsed_args.predict:
            predict_kwargs = {k: v for k, v in pipeline_args.items()
                             if k in ['data_path', 'model_path', 'output_path']}
            predictions = predict_next_12h(**predict_kwargs)
            results.update({'predictions_made': True, 'predictions': predictions})

        if parsed_args.evaluate:
            eval_kwargs = {k: v for k, v in pipeline_args.items()
                          if k in ['data_path', 'model_path', 'output_dir']}
            eval_results = evaluate_model(**eval_kwargs)
            results.update({'model_evaluated': True, 'evaluation_results': eval_results})

    # Print results summary
    print("\n" + "="*50)
    print("PIPELINE RESULTS")
    print("="*50)

    for key, value in results.items():
        if key == 'predictions':
            print(f"{key.title()}:")
            for campaign, pred in value.items():
                print(f"  {campaign}: ${pred:.2f}")
        elif key == 'training_metrics':
            print(f"{key.title()}:")
            for metric, val in value.items():
                if isinstance(val, float):
                    print(f"  {metric}: {val:.4f}")
        elif isinstance(value, dict):
            print(f"{key.title()}: {value}")
        else:
            print(f"{key.title()}: {value}")

    print("="*50)

    return results


if __name__ == "__main__":
    run_pipeline()
