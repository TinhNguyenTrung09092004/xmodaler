xmodaler.datasets 
=============================
.. autoclass:: xmodaler.datasets.DatasetFromList
    :members: __init__, __getitem__, __len__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.MapDataset
    :members: __init__, __getitem__, __len__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.MSCoCoDataset
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:

.. autoclass:: xmodaler.datasets.MSCoCoSampleByTxtDataset
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.MSCoCoBertDataset
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:
	
.. autofunction:: xmodaler.datasets.build_xmodaler_train_loader
	
.. autofunction:: xmodaler.datasets.build_xmodaler_valtest_loader

.. autofunction:: xmodaler.datasets.build_dataset_mapper
