from haven import haven_utils as hu
import itertools, copy
EXP_GROUPS = {}

EXP_GROUPS['pascal_point_level'] = hu.cartesian_exp_group({
                        'batch_size': 1,
                        'num_channels':1,
                        'dataset': [
                                {'name':'pascal'}
                                ],
                        'dataset_size':{'train':'all', 'val':'all'},
                        # 'dataset_size':{'train':10, 'val':10},
                        'max_epoch': [20],
                        'optimizer': [ "adam"],
                        'lr': [ 1e-5,],
                        'model': {'name':'semseg', 'loss':'point_level',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':21}
                        })

EXP_GROUPS['pascal_cross_entropy'] = hu.cartesian_exp_group({
                        'batch_size': 1,
                        'num_channels':1,
                        'dataset': [
                               {'name':'pascal'}
                                ],
                        'dataset_size':{'train':'all', 'val':'all'},
                        # 'dataset_size':{'train':10, 'val':10},
                        'max_epoch': [20],
                        'optimizer': [ "adam"],
                        'lr': [ 1e-5,],
                        'model': {'name':'semseg', 'loss':'cross_entropy',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':21}
                        })

EXP_GROUPS['pascal_consistency_loss'] = hu.cartesian_exp_group({
                        'batch_size': 1,
                        'num_channels':1,
                        'dataset': [
                               {'name':'pascal'}
                                ],
                        'dataset_size':{'train':'all', 'val':'all'},
                        #'dataset_size':{'train':10, 'val':10, 'test':10},
                        'max_epoch': [20],
                        'optimizer': [ "adam"],
                        'lr': [ 1e-5,],
                        'model': {'name':'semseg', 'loss':'consistency_loss',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':21}
                        })

EXP_GROUPS["pascal"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
       {'name':'pascal'},
    ],
    'dataset_size': [
        #  {'train':10, 'val':10, 'test':10},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [100],
    'optimizer': ["adam"],
    'lr': [1e-5, ],
    'model': [
        {'name': 'semseg', 'loss': 'rot_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},

        {'name': 'semseg', 'loss': 'cons_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},

        {'name': 'semseg', 'loss': 'point_level',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},

        {'name': 'semseg', 'loss': 'cross_entropy',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},
    ]
})

EXP_GROUPS["cityscapes"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
       {'name':'cityscapes'},
    ],
    'dataset_size': [
        #  {'train':10, 'val':10, 'test':10},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [100],
    'optimizer': ["adam"],
    'lr': [1e-5, ],
    'model': [
        {'name': 'semseg', 'loss': 'rot_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},

        {'name': 'semseg', 'loss': 'cons_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},

        {'name': 'semseg', 'loss': 'point_level',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},

        {'name': 'semseg', 'loss': 'cross_entropy',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},
    ]
})

EXP_GROUPS["weakly_JCUfish"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        {'name': 'JcuFish', 'n_classes': 2},
        # {'name': 'covid19_v2', 'n_classes': 2},
        # {'name':'covid19', 'n_classes':2},

    ],
    'dataset_size': [
        #  {'train':10, 'val':10, 'test':10},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [100],
    'optimizer': ["adam"],
    'lr': [1e-5, ],
    'model': [
         {'name': 'semseg', 'loss': 'rot_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 2},

        {'name': 'semseg', 'loss': 'cons_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 2},

        {'name': 'semseg', 'loss': 'point_level',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 2},

        {'name': 'semseg', 'loss': 'cross_entropy',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 2},
    ]
})

EXP_GROUPS["weakly_SUMfish"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        {'name': 'SumFish', 'n_classes': 2},
        # {'name': 'covid19_v2', 'n_classes': 2},
        # {'name':'covid19', 'n_classes':2},

    ],
    'dataset_size': [
         {'train':10, 'val':10, 'test':10},
        # {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [100],
    'optimizer': ["adam"],
    'lr': [1e-5, ],
    'model': [
         {'name': 'semseg', 'loss': 'rot_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 2},

        {'name': 'semseg', 'loss': 'cons_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 2},

        {'name': 'semseg', 'loss': 'point_level',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 2},

        {'name': 'semseg', 'loss': 'cross_entropy',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 2},
    ]
})

# ==========================
# Nuevos DeepAgro
# ==========================

EXP_GROUPS["weakly_JCUfish_aff"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        {'train':14, 'val':2, 'test':2},
        #{'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [40], #[1000],
    'optimizer': ["adam"],
    'lr': [1e-4], #[1e-4, 1e-5, 1e-6],
    'model':
    # [
    #     {'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      'with_attention':True,
    #      } for l in ['lcfcn_loss', ]] +
     [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         } for l in ['pseudo_mask', ]] +

    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         } for l in ['cross_entropy', ]] +


   [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         'with_affinity':True,
         } for l in ['cross_entropy', ]] +

    # [
    #     {'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      'with_affinity':True,
    #      'with_affinity_average':True,
    #      } for l in ['lcfcn_loss', ]] +

     [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         'with_affinity':True,
        #  'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]] +
         [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
        #  'with_affinity':True,
        #  'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]] 
         
         +
         [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         'with_affinity':True,
        #  'with_affinity_average':True,
         } for l in ['point_level', ]] +
         [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
        #  'with_affinity':True,
        #  'with_affinity_average':True,
         } for l in ['point_level', ]] 
        #  +

    # [
    #     {'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #     #  'with_affinity':True,
    #     #  'with_affinity_average':True,
    #      } for l in [ 'lcfcn_const_loss', 'lcfcn_loss', ]] +

    # +      [{'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      'with_affinity':True,
    #      'with_affinity_average':True,
    #      } for l in [ 'point_level', 'cons_point_loss']] #+

    #       [

    #     {'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      'with_affinity':True,
    #      } for l in [  'cons_point_loss', 'point_level',]]

    #      +
    #     [
    #      {'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      } for l in [ 'point_level', 'cons_point_loss']]

    ,
})

EXP_GROUPS["weakly_JCUfish_aff_OneModel"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        #{'train':60, 'val':10, 'test':24},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [30], #[1000],
    'optimizer': ["adam"],
    'lr': [1e-4], #0.5e-4, 1e-5], #[1e-4, 1e-5, 1e-6], # original con uno solo , ahi si funciona!!!!
    'model':
    # [
    #     {'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      'with_attention':True,
    #      } for l in ['lcfcn_loss', ]] +
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16', #'fcn8_vgg16',fcn8_resnet
         'n_channels': 3, 
         'n_classes': 2,
         'with_affinity':True,
         #'with_affinity_average':True,
         } for l in ['lcfcn_loss']] #cross_entropy

    # [
    #     {'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      'with_affinity':True,
    #      'with_affinity_average':True,
    #      } for l in ['lcfcn_loss', ]] +
    # [
    #     {'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #     #  'with_affinity':True,
    #     #  'with_affinity_average':True,
    #      } for l in [ 'lcfcn_const_loss', 'lcfcn_loss', ]] +

    # +      [{'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      'with_affinity':True,
    #      'with_affinity_average':True,
    #      } for l in [ 'point_level', 'cons_point_loss']] #+

    #       [

    #     {'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      'with_affinity':True,
    #      } for l in [  'cons_point_loss', 'point_level',]]

    #      +
    #     [
    #      {'name': 'semseg',
    #      'loss': l,
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      } for l in [ 'point_level', 'cons_point_loss']]

    #,
})

EXP_GROUPS["final_final"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        #{'train':60, 'val':30, 'test':8},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [200], #[1000],
    'optimizer': ["adam"],
    'lr': [1e-4], #[1e-4, 1e-5, 1e-6],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 1,
         'with_affinity':True,
         'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]]
})

#==========================
# Experiments for the .sh
# obs: basic with more max_epoch
EXP_GROUPS["weakly_aff_DeepAgro_exp1"] = hu.cartesian_exp_group({
    'batch_size': [10],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        #{'train':14, 'val':2, 'test':2},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [1000],
    'optimizer': ["adam"],
    'lr': [1e-4], #1e-5, 1e-6],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 1,
         'with_affinity':True,
         #'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]]  
})

# obs: mas opciones de lr y batch size
EXP_GROUPS["weakly_aff_DeepAgro_exp2"] = hu.cartesian_exp_group({
    'batch_size': [10],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        #{'train':14, 'val':2, 'test':2},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [1000],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 1,
         'with_affinity':True,
         #'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]]  
})

# obs: mas opciones de lr y with_affinity_average
EXP_GROUPS["weakly_aff_DeepAgro_exp3"] = hu.cartesian_exp_group({
    'batch_size': [10],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        #{'train':14, 'val':2, 'test':2},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [1000],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 1,
         'with_affinity':True,
         'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]]  
})

# obs: mas opciones de losses
EXP_GROUPS["weakly_aff_DeepAgro_exp4"] = hu.cartesian_exp_group({
    'batch_size': [10],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        #{'train':14, 'val':2, 'test':2},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [1000],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 1,
         'with_affinity':True,
         'with_affinity_average':True,
         } for l in ['lcfcn_loss', 'point_level', 'cons_point_loss']]
})

# obs: menos batch size y sin aff. multiples losses
EXP_GROUPS["weakly_aff_DeepAgro_exp5"] = hu.cartesian_exp_group({
    'batch_size': [5],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        #{'train':14, 'val':2, 'test':2},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [1000],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 1,
         'with_affinity':True,
         #'with_affinity_average':True,
         } for l in ['lcfcn_loss', 'point_level', 'cons_point_loss']]
})

# obs: multiples modelos con batch size de 1. 
EXP_GROUPS["weakly_aff_DeepAgro_exp6"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        #{'train':14, 'val':2, 'test':2},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [100],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model': [
        {'name': 'semseg', 'loss': 'rot_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 1},

        {'name': 'semseg', 'loss': 'cons_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 1},

        {'name': 'semseg', 'loss': 'point_level',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 1},

        {'name': 'semseg', 'loss': 'cross_entropy',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 1},

         {'name': 'semseg', 'loss': 'lcfcn_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 1},
    ]
})
# ==========================
# ==========================
# Nuevos test local
# ==========================
EXP_GROUPS["weakly_aff_test_exp1"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        #{'train':2, 'val':1, 'test':1},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [20],
    'optimizer': ["adam"],
    'lr': [1e-4], #, 1e-5, 1e-6],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 1,
         #'with_affinity':True,
         #'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]]  
})

# obs: mas opciones de lr y batch size
EXP_GROUPS["weakly_aff_test_exp2"] = hu.cartesian_exp_group({
    'batch_size': [10],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        {'train':2, 'val':1, 'test':1},
    ],
    'max_epoch': [2],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 1,
         'with_affinity':True,
         #'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]]  
})

# obs: mas opciones de lr y with_affinity_average
EXP_GROUPS["weakly_aff_test_exp3"] = hu.cartesian_exp_group({
    'batch_size': [10],
    'num_channels': 1,
    'dataset': [
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        {'train':2, 'val':2, 'test':2},
        #{'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [2],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 1,
         'with_affinity':True,
         'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]]  
})

# obs: mas opciones de losses
EXP_GROUPS["weakly_aff_test_exp4"] = hu.cartesian_exp_group({
    'batch_size': [10],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        {'train':2, 'val':2, 'test':2},
        #{'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [2],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 1,
         'with_affinity':True,
         'with_affinity_average':True,
         } for l in ['lcfcn_loss', 'point_level', 'cons_point_loss']]
})

# obs: menos batch size y sin aff. multiples losses
EXP_GROUPS["weakly_aff_test_exp5"] = hu.cartesian_exp_group({
    'batch_size': [5],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        #{'train':14, 'val':2, 'test':2},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [2],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 1,
         'with_affinity':True,
         #'with_affinity_average':True,
         } for l in ['lcfcn_loss', 'point_level', 'cons_point_loss']]
})

# obs: multiples modelos con batch size de 1. 
EXP_GROUPS["weakly_aff_test_exp6"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        #{'name': 'JcuFish', 'n_classes': 1},
        {'name': 'JcuFish', 'n_classes': 1},

    ],
    'dataset_size': [
        {'train':2, 'val':2, 'test':2},
        #{'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [2],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model': [
        {'name': 'semseg', 'loss': 'rot_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 1},

        {'name': 'semseg', 'loss': 'cons_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 1},

        {'name': 'semseg', 'loss': 'point_level',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 1},

        {'name': 'semseg', 'loss': 'cross_entropy',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 1},

         {'name': 'semseg', 'loss': 'lcfcn_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 1},
    ]
})

# ==========================
# the rest
def get_base_config(dataset_list, model_list, dataset_size={'train':'all', 'val':'all', 'test':'all'},
                             max_epoch=10):
        base_config =  {
                        'batch_size': 1,
                        'num_channels':1,
                        'dataset': [
                                {'name':name} for name in dataset_list
                                ],
                        'dataset_size':dataset_size,
                        'max_epoch': [max_epoch],
                        'optimizer': [ "adam"],
                        'lr': [ 1e-5,],
                        'model': model_list
                        }
        return base_config

pascal_baseline = get_base_config(
                                dataset_list=['pascal'],
                                dataset_size=[{'train':'all', 'val':'all'},],
                                model_list=[{'name':'semseg', 'loss':'point_level',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':21},
                                            {'name':'semseg', 'loss':'cross_entropy',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':21}])

EXP_GROUPS['pascal_baseline'] = hu.cartesian_exp_group(pascal_baseline)

pascal_baseline_debug = copy.deepcopy(pascal_baseline)
pascal_baseline_debug['dataset_size'] = [{'train':10, 'val':10},]
EXP_GROUPS['pascal_baseline_debug']  = hu.cartesian_exp_group(pascal_baseline_debug)


EXP_GROUPS["pascal_weakly"] = hu.cartesian_exp_group({
            'batch_size': [1],
            'num_channels': 1,
            'dataset': [
                {'name': 'pascal', 'n_classes': 2},

            ],
            'dataset_size': [
                 {'train':10, 'val':10, 'test':10},
                # {'train': 'all', 'val': 'all'},
            ],
            'max_epoch': [100],
            'optimizer': ["adam"],
            'lr': [1e-4, ],
            'model': [


                {'name': 'semseg', 'loss': 'point_loss',
                 'base': 'fcn8_vgg16',
                 'n_channels': 3, 'n_classes': 1},

    ]
})



EXP_GROUPS["cp_weakly"] = hu.cartesian_exp_group({
            'batch_size': [1],
            'num_channels': 1,
            'dataset': [
                {'name': 'cityscapes', 'n_classes': 2},

            ],
            'dataset_size': [
                 {'train':10, 'val':10, 'test':10},
                # {'train': 'all', 'val': 'all'},
            ],
            'max_epoch': [100],
            'optimizer': ["adam"],
            'lr': [1e-4, ],
            'model': [


                {'name': 'semseg', 'loss': 'point_loss',
                 'base': 'fcn8_vgg16',
                 'n_channels': 3, 'n_classes': 1},

    ]
})