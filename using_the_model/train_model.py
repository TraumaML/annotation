from simpletransformers.ner import NERModel, NERArgs
import pandas as pd
from pprint import pp


def main():

    train = pd.read_pickle(f"/data/batwood/R21-Modeling/improved_baseline/data/new_data/510_train.pkl")
    test = pd.read_pickle(f"/data/batwood/R21-Modeling/improved_baseline/data/new_data/510_dev.pkl")

    special_tokens_list =[
        "<Date>",
        "<alias>",
        "<clinic>",
        "<date 2015>",
        "<patient>",
        "<date>",
        "<first name <first name>",
        "<first name>",
        "<floor>",
        "<hospital>",
        "<hospitals>",
        "<hosptial>",
        "<last name>",
        "<location>",
        "<locaiton>",
        "<locartion>",
        "<location>",
        "<number>",
        "<organizatin>",
        "<organization in <location>",
        "<organization>",
        "<patient>",
        "<person10>",
        "<person1>",
        "<person2: <hospital>",
        "<person2>",
        "<person3>",
        "<person4>",
        "<person5>",
        "<person6>",
        "<person7>",
        "<person8>",
        "<person9>",
        "<phone number>",
        "<phone nunmber>",
        "<police department>",
        "<police>",
        "<prison>",
        "<school>",
        "<zip code>"
    ]
    model_path = "roberta-base"
    model_args = NERArgs()
    model_args.labels_list = list(set(train['labels']).union(set(test['labels'])))
    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "mcc"
    model_args.learning_rate = 0.00004
    model_args.batch_size = 16
    model_args.n_gpu = 1
    model_args.max_seq_length = 512
    model_args.save_eval_checkpoints= True
    model_args.save_model_every_epoch = True
    model_args.save_optimizer_and_scheduler = True
    model_args.save_best_model = True
    model_args.no_save = False
    model_args.num_train_epochs = 200
    model_args.early_stopping_metric_minimize = False
    model_args.special_tokens_list= special_tokens_list
    model_args.early_stopping_patience = 5
    model_args.evaluate_during_training_steps = 1000
    model_args.overwrite_output_dir = True

    model = NERModel('roberta', model_path, args=model_args, use_cuda=True)
    model.train_model(train)
    result, model_outputs, preds_list = model.eval_model(test)
    metrics = {"f1_score": float(result['f1_score']),
               "precision": float(result["precision"]),
               "recall": float(result["recall"])}
    
    pp(metrics)

if __name__ == "__main__":
    main()


