from lib.datasets import TopKCodes
from lib import models, trainers, predictors
from config import get_args_parser
import argparse

parser = argparse.ArgumentParser(parents=[get_args_parser()])
args = parser.parse_args()

method_dict = {
    'encoder': {
        'PQ': {
            'trainer': trainers.faiss.FaissPQTrainer, 
            'predictor': predictors.faiss.FaissPQPredictor
        }, 
        'RVQ': {
            'trainer': trainers.faiss.FaissRVQTrainer, 
            'predictor': predictors.faiss.FaissRVQPredictor
        }
    }, 
    'decoder': {
        'TFDec': {
            'model': models.both.TFDecVecCodeDecoder, 
            'trainer': {
                'AdamW-MSE': trainers.both.TFDecTrainerWithAdamWMSE
            }, 
            'predictor': predictors.both.TFDecPredictor
        }, 
        'TFEnc-code': {
            'model': models.code.TFEncCodeDecoder, 
            'trainer': {
                'AdamW-MSE': trainers.code.TFEncTrainerWithAdamWMSE
            }, 
            'predictor': predictors.code.TFEncPredictor
        }, 
        'TFEnc-vec': {
            'model': models.vec.TFEncVecDecoder, 
            'trainer': {
                'AdamW-MSE': trainers.vec.TFEncTrainerWithAdamWMSE
            }, 
            'predictor': predictors.vec.TFEncPredictor
        }, 
        'TFDec-code': {
            'model': models.code.CodeAttnDecoder, 
            'trainer': {
                'AdamW-MSE': trainers.code.TFDecTopKTrainerWithAdamWMSE
            }, 
            'predictor': predictors.code.TFDecTopKPredictor
        }, 
        'TFDec-code-top1q': {
            'model': models.code.CodeAttnDecoder, 
            'trainer': {
                'AdamW-MSE': trainers.code.TFDecTop1QTrainerWithAdamWMSE
            }, 
            'predictor': predictors.code.TFDecTop1QPredictor
        }, 
        'TFDec-vec': {
            'model': models.vec.VecAttnDecoder, 
            'trainer': {
                'AdamW-MSE': trainers.vec.TFDecTopKTrainerWithAdamWMSE
            }, 
            'predictor': predictors.vec.TFDecTopKPredictor
        }, 
        'TFDec-vec-top1q': {
            'model': models.vec.VecAttnDecoder, 
            'trainer': {
                'AdamW-MSE': trainers.vec.TFDecTop1QTrainerWithAdamWMSE
            }, 
            'predictor': predictors.vec.TFDecTop1QPredictor
        }
    }
}

if args.stage == 'train_encoder':
    trainer = method_dict['encoder'][args.encode_type]['trainer'](args)
    trainer.fit(args)

    predictor = method_dict['encoder'][args.encode_type]['predictor'](args)
    predictor.evaluate(args)
    predictor.save_codes(args)

elif args.stage == 'train_decoder':
    try:
        data = TopKCodes(
            args, 
            f"{args.code_dir}/{args.dataset}_{args.encode_type}_{args.codebooks}x{args.cb_bits}.npy", 
            f"{args.data_dir}/{args.dataset}.txt", 
            f"{args.cb_dir}/{args.dataset}_{args.encode_type}_{args.codebooks}x{args.cb_bits}.npy"
        )
    except:
        raise KeyError(f"Not support encode method: {args.encode_type}!")

    try:
        model = method_dict['decoder'][args.model]['model']
        trainer = method_dict['decoder'][args.model]['trainer'][f"{args.optimizer}-{args.loss_fn}"]
    except:
        raise KeyError(f"No model called: {args.model}!")

    if __name__ == '__main__':
        trainer = trainer(args, model)
        trainer.fit(args, data)

elif args.stage == 'eval':
    try:
        data = TopKCodes(
            args, 
            f"{args.code_dir}/{args.dataset}_{args.encode_type}_{args.codebooks}x{args.cb_bits}_query.npy", 
            f"{args.data_dir}/{args.dataset}_Query.txt", 
            f"{args.cb_dir}/{args.dataset}_{args.encode_type}_{args.codebooks}x{args.cb_bits}.npy"
        )
        orig_predictor = method_dict['encoder'][args.encode_type]['predictor']
    except:
        raise KeyError(f"Not support encode method: {args.encode_type}!")

    try:
        model = method_dict['decoder'][args.model]['model']
        predictor = method_dict['decoder'][args.model]['predictor']
    except:
        raise KeyError(f"No model called: {args.model}!")

    if __name__ == '__main__':
        orig_predictor = orig_predictor(args)
        origRecVecs = orig_predictor.predict(args)
        predictor = predictor(args, model)
        predictor.evaluate(args, data, origRecVecs)

else:
    raise ValueError(f"Wrong stage: {args.stage}, please input correct stage: --stage stage")