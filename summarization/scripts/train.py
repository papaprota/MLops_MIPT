import mlflow 
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from datamodule import NewsSummaryDataModule
from model import NewsSummaryModel

@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)
    

    mlf_logger = MLFlowLogger(experiment_name=cfg.mlflow.experiment_name,
                              tracking_uri=cfg.mlflow.uri)


    data_module = NewsSummaryDataModule(
        train_df=cfg.data.train,
        test_df=cfg.data.test,
        tokenizer=AutoTokenizer.from_pretrained(cfg.model.tokenizer_name),
        batch_size=cfg.training.batch_size,
        text_max_token_len=cfg.model.text_max_token_len,
        summary_max_token_len=cfg.model.summary_max_token_len,
        num_workers=cfg.model.num_workers
    )
    

    model = NewsSummaryModel(cfg.model.model_name,
                             lr=cfg.model.lr,
                             eps=cfg.model.eps)


    checkpoint_callback = ModelCheckpoint(
    dirpath=cfg.model.dirpath,
    filename=cfg.model.filename,
    save_top_k=cfg.training.save_top_k,
    verbose=True,
    monitor=cfg.training.monitor,
    mode=cfg.training.mode
)   

    mlf_logger.log_hyperparams(dict(model.hparams))

    trainer = pl.Trainer(
        logger=mlf_logger,
        callbacks=checkpoint_callback,
        max_epochs=cfg.training.epochs,
        devices=cfg.training.devices,
        accelerator=cfg.training.accelerator,
        log_every_n_steps=cfg.training.log_every_n_steps,
        precision=cfg.training.precision,
    )

    
    mlflow.pytorch.autolog()

    with mlflow.start_run(run_id=mlf_logger.run_id):
        trainer.fit(model, data_module)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",            
            registered_model_name=cfg.model.name,
        )

if __name__ == "__main__":
    train()