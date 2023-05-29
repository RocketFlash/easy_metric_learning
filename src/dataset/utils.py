def get_labels_to_ids_map(labels):
    labels_to_ids = {}
    ids_to_labels = {}
    idx = 0

    for label in labels:
        if label not in labels_to_ids:
            labels_to_ids[label] = idx
            ids_to_labels[idx] = label
            idx+=1
    
    return labels_to_ids, ids_to_labels