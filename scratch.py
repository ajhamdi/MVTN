    elif setup["visualize_verts_mode"]:
        from tqdm import tqdm

        list_dict = ListDict(["sample_nb","nb_verts","class_label"])
        print("visualizing train set")
        for ii, (targets, meshes, points) in enumerate(tqdm(dset_train)):
            nb_verts = meshes.verts_padded().shape[1]
            list_dict.append(
                {"sample_nb": ii, "nb_verts": nb_verts, "class_label": targets})
        save_results(os.path.join(setup["verts_dir"],
                                "train_SMPLmesh_verts.csv"), list_dict)

        list_dict = ListDict(["sample_nb", "nb_verts", "class_label"])
        print("visualizing test set")
        for ii, (targets, meshes, points) in enumerate(tqdm(dset_val)):
            nb_verts = meshes.verts_padded().shape[1]
            list_dict.append(
                {"sample_nb": ii, "nb_verts": nb_verts, "class_label": targets})
        save_results(os.path.join(setup["verts_dir"], "test_SMPLmesh_verts.csv"), list_dict)



    if setup["extract_features_mode"]:
        from tqdm import tqdm
        torch.multiprocessing.set_sharing_strategy('file_system')
        if setup["shape_extractor"] == "PointNet":
            point_network = PointNet(40, alignment=True).cuda()
        elif setup["shape_extractor"] == "DGCNN":
            point_network = SimpleDGCNN(40).cuda()
        point_network.eval()
        load_point_ckpt(
            point_network,  setup["shape_extractor"],  ckpt_dir='./checkpoint')

        print('\nEvaluation pointnetwork alone:')
        avg_train_acc, avg_train_loss = test_point_network(
            point_network, criterion, data_loader=train_loader, )
        avg_test_acc, avg_loss = test_point_network(
            point_network, criterion, data_loader=val_loader, )
        print('\ttrain acc: %.2f - train Loss: %.4f' %
            (avg_train_acc.item(), avg_train_loss.item()))
        print('\tVal Acc: %.2f - val Loss: %.4f' %
            (avg_test_acc.item(), avg_loss.item()))

        print('\Extracting the point features:')
        train_points_features_list = [file_name.replace(".off", "_PFeautres.pkl") for file_name in dset_train.meshes_list if file_name[-4::] == ".off"]
        test_points_features_list = [file_name.replace(".off", "_PFeautres.pkl") for file_name in dset_val.meshes_list if file_name[-4::] == ".off"]

        for i, (_, _, _, points) in enumerate(tqdm(dset_train)):
            with torch.no_grad():
                points = points[None,...].transpose(1, 2).cuda()
                logits, post_max, transform_matrix = point_network(points)
                saveables = {'logits': logits.cpu().numpy(),
                            'post_max': post_max.cpu().numpy(),
                            "transform_matrix": transform_matrix.cpu().numpy(),
                            }
                save_obj(saveables, train_points_features_list[i])
        print("finished train set")

        for i, (_, _, _, points) in enumerate(tqdm(dset_val)):
            with torch.no_grad():
                points = points[None, ...].transpose(1, 2).cuda()
                logits, post_max, transform_matrix = point_network(points)
                saveables = {'logits': logits.cpu().numpy(),
                            'post_max': post_max.cpu().numpy(),
                            "transform_matrix": transform_matrix.cpu().numpy(),
                            }
                save_obj(saveables, test_points_features_list[i])
        print("finished test set")

   elif setup["late_fusion_mode"]:
        from tqdm import tqdm
        RESNET_FEATURE_SIZE = 40
        models_bag["mvnetwork"].eval()
        models_bag["mvtn"].eval()
        models_bag["mvrenderer"].eval()
        # models_bag["feature_extractor"].eval()
        if setup["log_metrics"]:
            writer = SummaryWriter(setup["logs_dir"])
        torch.multiprocessing.set_sharing_strategy('file_system')
        if setup["shape_extractor"] == "PointNet":
            point_network = PointNet(40, alignment=True).cuda()
        elif setup["shape_extractor"] == "DGCNN":
            point_network = SimpleDGCNN(40).cuda()
        point_network.eval()
        load_point_ckpt(
            point_network,  setup["shape_extractor"],  ckpt_dir='./checkpoint')

        print('\nEvaluation pointnetwork alone:')
        # all_imgs_list = [640,669,731,2100,2000]
        # all_imgs_list = [2529]

        # all_imgs_list = [2438,  2439,  520,  2573,  527,  2448,  2447,  2449,  2575,  534,  152,  2586,  2458,  2464,  39,  2472,  2425,  426,427,              431,                     47,                     51,      2487,      2489,      58,      2492,      2493,      450,      579,      68,      2501,      73,      2514,      212,      469,      2525,                     94,                     93,                     2529,                     2535,                     2536,                     487,                     234,                     2539,                     237,                     2418,                     505]
        all_imgs_list = [118, 119, 120, 251, 260, 269, 323, 355, 468, 479, 607, 620, 673, 711, 713, 715, 723, 740, 751, 759, 782, 783, 788, 791, 800, 812, 816, 856, 878, 882, 886, 888, 891, 908, 926, 927, 942, 943, 944, 945, 946, 947, 948, 951, 953, 955, 956, 957, 958, 960, 961, 972, 1033, 1057, 1186, 1187, 1195, 1270, 1299, 1303, 1344, 1424, 1428, 1444, 1457, 1468, 1473, 1475, 1478, 1498,
                        1499, 1504, 1506, 1532, 1550, 1568, 1587, 1594, 1633, 1645, 1648, 1651, 1655, 1677, 1712, 1746, 1749, 1776, 1821, 1853, 1859, 1867, 1868, 1869, 1923, 1988, 1993, 1996, 2000, 2001, 2026, 2042, 2050, 2053, 2065, 2089, 2114, 2115, 2295, 2312, 2314, 2322, 2330, 2332, 2335, 2336, 2337, 2342, 2360, 2375, 2380, 2381, 2382, 2386, 2389, 2391, 2395, 2400, 2405, 2409, 2416, 2420, 2429, 2441]
        # all_imgs_list = list(range(len(dset_val)))
        # all_imgs_list = [9,17,24,27,31,65,70,94,102,109,112,114,115,132,139,141,143,144,147,148,176,223,228,252,257,259,271,274,276,287,289,290,295,298,301,306,310,312,315,317,318,319,322,330,332,340,346,347,348,353,355,356,357,358,359,364,365,367,369,378,380,384,386,387,391,393,423,436,438,455,456,464,465,467,471,476,477,479,480,481,483,486,488,490,492,493,494,496,499,500,501,506,511,515,516,517,519,520,522,523,524,530,534,535,536,538,540,543,545,546,550,558,561,562,565,570,571,572,573,574,578,580,581,584,585,587,593,595,596,611,612,617,630,637,638,643,644,649,655,684,686,693,694]
        visualize_retrieval_views(dset_val, all_imgs_list,
                                models_bag, setup,)
        compiled_analysis_list =  analyze_rendered_views(dset_val, all_imgs_list,models_bag, setup, device)
        f = open('test_avg_pos.txt', 'w')
        f.writelines(["{:.3f} \n".format(x) for x in compiled_analysis_list])
        f.close()
        avg_train_acc, avg_train_loss = test_point_network(
            point_network, criterion, data_loader=train_loader, setup=setup, )
        avg_test_acc, avg_loss = test_point_network(
            point_network, criterion, data_loader=val_loader, setup=setup, )
        print('\ttrain acc: %.2f - train Loss: %.4f' %
              (avg_train_acc.item(), avg_train_loss.item()))
        print('\tVal Acc: %.2f - val Loss: %.4f' %
              (avg_test_acc.item(), avg_loss.item()))
        raise Exception("just checking the visualization")


if setup["measure_speed_mode"]:

    models_bag["mvnetwork"].eval()
    models_bag["mvtn"].eval()
    models_bag["mvrenderer"].eval()
    # models_bag["feature_extractor"].eval()
    if "modelnet" not in setup["data_dir"].lower():
        raise Exception('Occlusion is only supported froom ModelNet now ')
    from tqdm import tqdm
    torch.multiprocessing.set_sharing_strategy('file_system')

    print('\Evaluatiing speed memory  :')
    print("network", "\t", "MACs", "\t", "# params", "\t", "time/sample (ms)")
    override = True
    MAX_ITER = 10000
    for network in ["MVTN", "PointNet", "DGCNN", "MVCNN"]:
        if network == "PointNet":
            setup["shape_extractor"] = "PointNet"
            point_network = PointNet(40, alignment=True).cuda()
        elif network == "DGCNN":
            setup["shape_extractor"] = "DGCNN"
            point_network = SimpleDGCNN(40).cuda()
        if network in ["DGCNN", "PointNet"]:
            point_network.eval()
            load_point_ckpt(point_network,  setup["shape_extractor"],
                            ckpt_dir='./checkpoint', verbose=False)
            macs, params = get_model_complexity_info(
                point_network, (3, setup["nb_points"]), as_strings=True, print_per_layer_stat=False, verbose=False)
            inp = torch.rand((1, 3, setup["nb_points"])).cuda()
            avg_time = profile_op(MAX_ITER, point_network, inp)
        elif network in ["MVCNN", "ViewGCN"]:
            macs, params = get_model_complexity_info(
                models_bag["mvnetwork"], (setup["nb_views"], 3, setup["image_size"], setup["image_size"]), as_strings=True, print_per_layer_stat=False, verbose=False)
            inp = torch.rand(
                (1, setup["nb_views"], 3, setup["image_size"], setup["image_size"])).cuda()
            avg_time = profile_op(MAX_ITER, models_bag["mvnetwork"], inp)
        else:
            macs, params = get_model_complexity_info(
                models_bag["mvtn"], (setup["features_size"],), as_strings=False, print_per_layer_stat=False, verbose=False)
            inp = torch.rand((1, setup["features_size"])).cuda()
            avg_time = profile_op(MAX_ITER, models_bag["mvtn"], inp)
        print(network, "\t", macs, "\t", params,
              "\t", "{}".format(avg_time*1e3))
