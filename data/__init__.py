from data.benchmark import BenchmarkDataset
from torch.utils.data import DataLoader
from utils import AttrDict

import logging
logger = logging.getLogger(__name__)

def _get_test_dataloaders(args, degradation_operation=None):
    dataloaders = []
    for d in args.data.data_test:
        if d in ['Set14', 'B100', 'Urban100', 'DIV2K']:
            test_dataset = BenchmarkDataset(args, name=d, degradation_operation=degradation_operation)
        else:
            raise NotImplementedError()

        dataloaders.append(DataLoader(test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.data.n_threads,
                        pin_memory=not args.sys.cpu))
    return dataloaders

def get_dataloader(args):
    train_dataloader = None
    test_dataloaders = {}

    if not args.train.test_only:
        assert args.data.data_train == 'DIV2K'
        from data.train_dataset import DIV2KDataset
        train_dataset = DIV2KDataset(args)
        
        bs = args.optim.task_batch_size
 
        train_dataloader = DataLoader(train_dataset,
            batch_size=bs,
            shuffle=True,
            pin_memory=not args.sys.cpu,
            num_workers=args.data.n_threads)

    if args.data.custom_data_path is not None:
        logger.info('[*] Evaluating on custom images')
        from data.custom import CustomDataset
        test_dataset = CustomDataset(args, 
                        data_path=args.data.custom_data_path)

        test_dataloaders['custom'] = [DataLoader(test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.data.n_threads,
                pin_memory=not args.sys.cpu)]
    elif args.data.data_test is not None:
        logger.info('[*] Evaluating on benchmarks')
        if type(args.degradation_operations.test) != list:
            args.degradation_operations.test = [args.degradation_operations.test]
    
        for degradation_operation in args.degradation_operations.test:
            degradation_operation = AttrDict(degradation_operation)
            assert hasattr(degradation_operation, 'name')
            test_dataloaders[degradation_operation.name] = _get_test_dataloaders(args, degradation_operation)


    return train_dataloader, test_dataloaders        
