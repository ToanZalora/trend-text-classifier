import boto3
import pandas as pd
import io
import os

#-#-#-#-#----CLASS----#-#-#-#-#	
class Storage:
    def read_csv(self, 
	             filename: str):
        pass
		
    def write_csv(self, 
	              data, 
				  filename: str):
        pass
		
    def exists(self, 
	           filename: str) -> bool:
        return False
		
    def makedirs(self, 
	             path: str):
        pass
		
#-#-#-#-#----CLASS----#-#-#-#-#		
class S3(Storage):
    def __init__(self, 
	             bucket    : str, 
				 key_id    : str, 
				 secret_key: str):
        self.bucket = bucket
        self.s3     = boto3.client('s3', 
		                            aws_access_key_id     = key_id,  
									aws_secret_access_key = secret_key)
    
    def read_csv(self, 
	             filename: str):
        obj         = self.s3.get_object(Bucket = self.bucket, 
		                                 Key    = filename)
        return pd.read_csv(io.BytesIO(obj['Body'].read()))

    def write_csv(self, 
	              data, 
				  filename: str):
        buffer      = io.StringIO()
        data.to_csv(buffer, 
		            encoding = 'utf_8', 
					index    = False
					)
        self.s3.put_object(Bucket = self.bucket, 
		                   Key    = filename, 
						   Body   = buffer.getvalue()
						   )

    def exists(self, filename: str) -> bool:
        response    = self.s3.list_objects_v2(Bucket  = self.bucket, 
		                                      MaxKeys = 1, 
											  Prefix  = filename)
        return response['KeyCount'] > 0

    def makedirs(self, path: str):
        pass


#-#-#-#-#----CLASS----#-#-#-#-#
class LocalFS(Storage):
    def __init__(self, 
	             root = None):
        if root:
            os.makedirs(root, 
			            exist_ok = True)
	
    def read_csv(self, 
	             filename: str) -> pd.DataFrame:
        return pd.read_csv(filename, 
		                   encoding = 'latin-1')
	
    def write_csv(self, 
	              data, 
				  filename: str):
        data.to_csv(filename, 
		            encoding = 'utf_8', 
					index    = False)
	
    def exists(self, 
	           filename: str) -> bool:
        return os.path.isfile(filename)
		    
    def makedirs(self, 
	             path: str):
        os.makedirs(path, 
		            exist_ok = True)

					
#-#-#-#-#----CLASS----#-#-#-#-#			
def create_storage(config) -> Storage:
    if config.runtime.storage == 's3':
        return S3(config.server.s3.bucket, 
		          config.server.s3.key_id, 
				  config.server.s3.secret_key
				  )
				  
    os.makedirs(config.paths.raw_data_folder, exist_ok = True)
    os.makedirs(config.paths.input_folder,    exist_ok = True)
    return LocalFS()