# How to develop a new routine

1. Create a specific python file with the name of the routine, like: `register_local.py`

2. Inside this file, create a class with the name of the routine, in PascalCase: `RegisterLocal` , this class inherits from `Routine` class: 

   ```python
   from chromapylot.routines.routine import Routine
   
   class RegisterLocal(Routine):
   ```

3. Initialize the routine with a `DataManager` to help the routine to load and save data and log every thing:

   
   ```python
   # ...
   from chromapylot.core.data_manager import DataManager
   
   class RegisterLocal(Routine):
       def __init__(self, data_manager: DataManager):
           super().__init__(data_manager=data_manager)
   ```


4. Define the parameter section(s) that the routine required, can be one or more. the param section should be a type of a class present in the parameters folder.

   ```python
   # ...
   from chromapylot.parameters.registration_params import RegistrationParams
   
   class RegisterLocal(Routine):
       def __init__(
           self,
           data_manager: DataManager,
           registration_params: RegistrationParams,
       ):
           super().__init__(data_manager=data_manager)
   ```

5. A sub-class of Routine have two mandatory data types (`DataType`) for the input and the output and two optional types for a reference and a supplementary data. The reference data are just used like a "read mode" inside the routine and it's common for all files to process inside a same analysis type. Supplementary data is also in a "read only mode" but it's specific for each file.

   ```python
   # ...
   from chromapylot.core.core_types import DataType
   
   class RegisterLocal(Routine):
       def __init__(
           self,
           data_manager: DataManager,
           registration_params: RegistrationParams,
       ):
           super().__init__(
               data_manager=data_manager,
               input_type=DataType.IMAGE_3D_SHIFTED,
               output_type=DataType.REGISTRATION_TABLE,
               reference_type=DataType.IMAGE_3D,
               supplementary_type=None,
           )
   ```

   

6. Now you have to overwrite each generic methods that you need for your routine:

   ```python
   #
   from typing import List
   # ...
   
   	def load_data(self, input_path):
           raise NotImplementedError
           
    def load_reference_data(self, paths: List[str]):
        raise NotImplementedError

    def load_supplementary_data(self, input_path, cycle):
        raise NotImplementedError
        
    def run(self, data, supplementary_data=None):
        raise NotImplementedError

    def save_data(self, data, input_path, input_data, supplementary_data):
        raise NotImplementedError
   
   ```

   

7. There is a final template to start:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from chromapylot.core.core_types import DataType
from chromapylot.core.data_manager import DataManager
from chromapylot.routines.routine import Routine
from chromapylot.parameters.registration_params import RegistrationParams


class RegisterLocal(Routine):
    def __init__(
        self,
        data_manager: DataManager,
        registration_params: RegistrationParams,
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_3D_SHIFTED,
            output_type=DataType.REGISTRATION_TABLE,
            reference_type=DataType.IMAGE_3D,
            supplementary_type=None,
        )
        self.dirname = "register_local"
        self.example_name = registration_params.example_name

    def load_data(self, input_path):
        raise NotImplementedError

    def load_reference_data(self, paths: List[str]):
        raise NotImplementedError

    def load_supplementary_data(self, input_path, cycle):
        raise NotImplementedError

    def run(self, data, supplementary_data=None):
        raise NotImplementedError

    def save_data(self, data, input_path, input_data, supplementary_data):
        raise NotImplementedError

```