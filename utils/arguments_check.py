def arguments_check_func(args):
    if args.model=="unrolled_lrp":
        if args.ablation_test ==False:
            args.iterative_gradients=True
            args.add_classification=True
            args.semisup_dataset=True
            print(f"Warning: args.iterative_gradients, args.add_classificatio and args.semisup_dataset are set to True automatically")
            print(f"If you need to change this setting, please go to the file utils/arguments_check.py")
   
        else:
            # ablation test
            # if normal_deconv is set to True, the normal_relu is set to True automatically
            # if args.normal_deconv and not args.normal_unpool:
            #     args.normal_relu=True
            #     args.iterative_gradients=False
            #     # args.remove_heaviside=False
            #     print("The argument normal_relu=True and iterative_gradients=False automatically")
            #     print("Unrolled_lrp model's ablation test2")
            

            if args.normal_unpool:
                args.normal_relu=True
                args.normal_deconv=True
                args.iterative_gradients=False
                # args.remove_heaviside=False
                print("The argument normal_relu and args.normal_deconv=True and iterative_gradients=False automatically")
                print("Unrolled_lrp model's ablation test3")

            if args.normal_relu== True and args.normal_deconv==False:
                print("Unrolled_lrp model's ablation test1")

            if (args.normal_relu== False and args.normal_deconv==False) or (args.normal_deconv==True):
                args.multiply_input = False

    return 