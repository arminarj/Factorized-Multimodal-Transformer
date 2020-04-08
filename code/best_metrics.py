class Best():
    best_epoch = 0

    max_train_f1 = 0
    max_test_f1 = 0
    max_valid_f1 = 0

    max_valid_f1_mfn = 0
    max_valid_f1_raven = 0
    max_valid_f1_muit = 0

    max_test_f1_mfn = 0
    max_test_f1_raven = 0
    max_test_f1_muit = 0

    max_test_prec = 0
    max_valid_prec = 0
    max_train_prec = 0
    max_train_recall = 0
    max_test_recall = 0
    max_valid_recall = 0
    max_train_acc = 0
    max_valid_acc = 0
    max_test_acc = 0
    max_valid_ex_zero_acc = 0
    max_test_ex_zero_acc = 0
    max_valid_acc_5 = 0
    max_test_acc_5 = 0
    max_valid_acc_7 = 0
    max_test_acc_7 = 0

    test_acc_at_valid_max = 0
    test_ex_zero_acc_at_valid_max = 0
    test_acc_5_at_valid_max = 0
    test_acc_7_at_valid_max = 0
    test_f1_at_valid_max = 0

    test_f1_mfn_at_valid_max = 0
    test_f1_raven_at_valid_max = 0
    test_f1_muit_at_valid_max = 0

    test_prec_at_valid_max = 0
    test_recall_at_valid_max = 0

    min_train_mae = 10
    min_test_mae = 10
    max_test_cor = 0
    min_valid_mae = 10
    max_valid_cor = 0
    test_mae_at_valid_min = 10
    test_cor_at_valid_max = 0

    iemocap_emos = ["Neutral", "Happy", "Sad", "Angry"]
    split = ['valid', 'test_at_valid_max', 'test']
    max_iemocap_f1 = {}
    max_iemocap_acc = {}
    for sp in split:
        max_iemocap_f1[sp] = {}
        max_iemocap_acc[sp] = {}
        for em in iemocap_emos:
            max_iemocap_f1[sp][em] = 0
            max_iemocap_acc[sp][em] = 0

    pom_cls = ["Confidence", "Passionate", "Voice pleasant", "Dominant", "Credible", "Vivid", "Expertise",
               "Entertaining", "Reserved", "Trusting", "Relaxed", "Outgoing", "Thorough",
               "Nervous", "Sentiment", "Persuasive", "Humorous"]

    max_pom_metrics = {metric: {} for metric in ['acc', 'corr']}
    for metric in ['acc', 'corr']:
        for sp in split:
            max_pom_metrics[metric][sp] = {}
            for cls in pom_cls:
                max_pom_metrics[metric][sp][cls] = 0

    best_pom_mae = {}
    for sp in split:
        best_pom_mae[sp] = {}
        for cls in pom_cls:
            best_pom_mae[sp][cls] = 10
