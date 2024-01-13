from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_uri: str = "./data/external/chest_xray/train"
    print("train_data_uri", train_data_uri)
    test_data_uri: str = "./data/external/chest_xray/test"
    print("test_data_uri", test_data_uri)
    # validation_data_uri: str = f"s3://{app_config.storage.bucket_name}/{app_config.storage.files.validation_data}"
    # print("validation_data_uri", validation_data_uri)


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        return (
            self.ingestion_config.train_data_uri,
            self.ingestion_config.test_data_uri
        )
