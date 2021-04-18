
def from_args_to_string(args):
    # create export dir if it doesnt exist
    directory = "{}".format(args.training_dataset)
    directory += "_{}".format(args.arch)
    directory += "_{}".format(args.pool)
    if args.local_whitening:
        directory += "_lwhiten"
    if args.regional:
        directory += "_r"
    if args.whitening:
        directory += "_whiten"
    if not args.pretrained:
        directory += "_notpretrained"
    if args.test_whiten:
        directory += "_test_whiten_on_{}".format(args.test_whiten)
    directory += "_{}_m{:.2f}".format(args.loss, args.loss_margin)
    directory += "_{}_lr{:.1e}_wd{:.1e}".format(args.optimizer, args.lr, args.weight_decay)
    directory += "_nnum{}_qsize{}_psize{}".format(args.neg_num, args.query_size, args.pool_size)
    directory += "_bsize{}_uevery{}_imsize{}".format(args.batch_size, args.update_every, args.image_size)
    directory += "_temp{}".format(args.temp)
    directory += "_{}".format(args.mode)
    if args.mode == "ap":
        directory += "_nexamples_{}".format(args.nexamples) 
    #if args.ts:
    #    directory += "_ts"
    #if args.reg:
    #    directory += "_reg"
    directory += "_teach_{}".format(args.teacher)
    directory += args.comment
    return directory