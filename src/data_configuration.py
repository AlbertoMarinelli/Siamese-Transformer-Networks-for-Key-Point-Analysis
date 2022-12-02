import pandas as pd

def configure_KPA_2021_shared_task (args_path, kps_path, labels_path, type_name='generic', merge=False, neg_value=0, pos_value=1):

    # read the data from original datasets provided in the github repo for the task
    df_arguments = pd.read_csv(args_path)[['arg_id', 'argument']]
    df_key_points = pd.read_csv(kps_path)[['key_point_id', 'key_point']]
    df_labels = pd.read_csv(labels_path)

    # arrays for data
    labels = [None] * len(df_labels)  # labels
    args = [None] * len(df_labels)  # arguments
    kps = [None] * len(df_labels)  # key points

    # fill the arrays.
    for i in range(len(df_labels)):
        label = df_labels.iloc[i]
        args[i] = df_arguments[df_arguments['arg_id'] == label['arg_id']]['argument'].array[0]
        kps[i] = df_key_points[df_key_points['key_point_id'] == label['key_point_id']]['key_point'].array[0]

    # in this setting we want minimise the cosine similarity of encodings of related sentences,
    # the desired otput will be values in the domain of the cosine-similarity function:
    # cos. sim. = -1  if two vectors (encodings) are different,
    #           = 1 if two vectors are similar
    labels = list(map(lambda x: neg_value if x == 0 else pos_value, df_labels['label']))
    if merge:
        args = pd.DataFrame(args)
        kps = pd.DataFrame(kps)
        labels = pd.DataFrame(labels)
        args.rename(columns={0: "args"}, inplace=True)
        args["kps"]=kps
        args["labels"]=labels
        args.to_csv('../dataset_KPA_2021/dataset_' + type_name + '.csv', index=False)
    else:
        args = pd.DataFrame(args)
        args.to_csv('../dataset_KPA_2021/args_'+type_name+'.csv', index=False)

        kps = pd.DataFrame(kps)
        kps.to_csv('../dataset_KPA_2021/kps_'+type_name+'.csv', index=False)

        labels = pd.DataFrame(labels)
        labels.to_csv('../dataset_KPA_2021/labels_'+type_name+'.csv', index=False)