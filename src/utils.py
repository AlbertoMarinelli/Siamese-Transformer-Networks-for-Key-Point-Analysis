from SBERT_models import build_model
import pandas as pd
import json

def compute_prec(path):

  preds_dev = pd.read_csv(path)
  labels = pd.read_csv('../kpm_data/labels_dev.csv')

  print(len(preds_dev), len(labels))

  jsons = {}
  for i, line in preds_dev.T.iteritems():
    pred = line[0]
    line_labels = labels.iloc[i]
    arg_id = line_labels['arg_id']
    kp_id = line_labels['key_point_id']
    
    try:
      jsons[arg_id][kp_id] = pred
    except KeyError:
      jsons[arg_id] = {kp_id:pred}

  
  with open('preds_dev.json', 'w', encoding='utf-8') as f:
      json.dump(jsons, f, ensure_ascii=False, indent=4)

  from track_1_kp_matching import load_kpm_data, get_predictions, evaluate_predictions

  gold_data_dir = '../kpm_data/'
  predictions_file = 'preds_dev.json'

  arg_df, kp_df, labels_df = load_kpm_data(gold_data_dir, subset="dev")

  merged_df = get_predictions(predictions_file, labels_df, arg_df, kp_df)
  mAP_strict, mAP_relaxed = evaluate_predictions(merged_df)
  
  return mAP_strict, mAP_relaxed

def grid_search_iteration(config, dataset_train, batch_size, inputs_dev, num_tests = 1, emb='bert-base-uncased'):
  
  sum_maps_strict = 0
  sum_maps_relaxed = 0

  for i in range(num_tests):
    # build the model
    model = build_model(config, embeddings=emb)
    
    # train the model
    h = model.fit(dataset_train.batch(batch_size), epochs = config['num_epochs'])

    # compute predictions
    preds_dev = model.predict(inputs_dev)

    try:
      #save predictions
      path = f'./preds_dev_{config["cls_token_activate"]}_{config["num_epochs"]}_{config["lr"][0]}_{config["lr"][1]}.csv'
    
    except:
      #save predictions
      path = f'./preds_dev_{config["num_epochs"]}_{config["lr"][0]}_{config["lr"][1]}.csv'
    
    pd.DataFrame(preds_dev).to_csv(path,index=False)

    # compute precision
    mAP_strict, mAP_relaxed = compute_prec(path)

    sum_maps_strict +=  mAP_strict
    sum_maps_relaxed +=  mAP_relaxed
  
  mean_map_strict = sum_maps_strict / num_tests
  mean_map_relaxed = sum_maps_relaxed / num_tests

  return f' {config} {mean_map_strict, mean_map_relaxed}'