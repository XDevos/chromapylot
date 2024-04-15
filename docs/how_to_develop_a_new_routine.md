# How to develop a new routine

1. Create a specific python file with the name of the routine, like: `register_local.py`

2. Inside this file, create a class with the name of the routine, in PascalCase: `RegisterLocal` , this class inherits from `Module` class: 

   ```python
   class RegisterLocal(Module):
   ```

3. Define the parameter section(s) that the routine required, can be one or more. the param section should be a type of a class present in the parameters folder.

   ```python
   from chromapylot.modules.module import Module
   from chromapylot.parameters.registration_params import RegistrationParams
   
   
   class RegisterLocal(Module):
       def __init__(self, registration_params: RegistrationParams):
   ```

4. A sub-class of Module have two mandatory data types (`DataType`) for the input and the output and two optional types for a reference and a supplementary data. The reference data are just used like a "read mode" inside the routine and it's common for all files to process inside a same analysis type. Supplementary data is also in a "read only mode" but it's specific for each file.

   ```python
   class RegisterLocal(Module):
       def __init__(self, registration_params: RegistrationParams):
           super().__init__(
               input_type=DataType.IMAGE_3D_SHIFTED,
               output_type=DataType.REGISTRATION_TABLE,
               reference_type=DataType.IMAGE_3D,
               supplementary_type=None,
           )
   ```

   

5. 

