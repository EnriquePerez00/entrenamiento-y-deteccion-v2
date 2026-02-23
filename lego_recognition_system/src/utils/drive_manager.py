import os
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']

class DriveManager:
    def __init__(self, credentials_path='credentials.json', token_path='token.pickle'):
        """
        Initialize Drive Manager.
        credentials_path: Path to the OAuth 2.0 Client ID JSON file.
        token_path: Path to store the user's access and refresh tokens.
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self.creds = None
        self.account_email = "Unknown"

    def authenticate(self):
        """Authenticates the user and builds the Drive service."""
        creds = None
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing token: {e}. Re-authenticating...")
                    creds = None
            else:
                # If expired but no refresh token, or just no creds at all
                creds = None
            
            if not creds:
                if not os.path.exists(self.credentials_path):
                    # Fallback to default credentials.json if specified path doesn't exist
                    if os.path.exists("credentials.json"):
                        self.credentials_path = "credentials.json"
                    else:
                        raise FileNotFoundError(f"Credentials file not found at {self.credentials_path}")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                # Ensure we get a refresh token
                creds = flow.run_local_server(port=8080, access_type='offline', prompt='consent')
            
            # Save the credentials for the next run
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)

        self.creds = creds
        self.service = build('drive', 'v3', credentials=creds)
        
        # Get User Info
        try:
             about = self.service.about().get(fields="user").execute()
             self.account_email = about['user']['emailAddress']
             print(f"✅ Authenticated as: {self.account_email}")
        except:
             pass
             
        return True

    def ensure_folder(self, folder_name, parent_id=None):
        """Checks if a folder exists, creates it if not."""
        if not self.service: self.authenticate()
        
        # If no parent_id is specified, default to the Drive 'root' (MyDrive)
        # to avoid finding/creating folders with the same name anywhere in Drive.
        if not parent_id:
             parent_id = 'root'
             
        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
        query += f" and '{parent_id}' in parents"
            
        results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        items = results.get('files', [])
        
        if items:
            return items[0]['id']
        else:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]
            }
                
            file = self.service.files().create(body=file_metadata, fields='id').execute()
            print(f"Created folder '{folder_name}' ({file.get('id')}) on {self.account_email}")
            return file.get('id')

    def upload_file(self, local_path, folder_id, remote_name=None, overwrite=True, check_date=True):
        """
        Uploads a file to the specified folder ID.
        check_date: If True, only uploads if local file is newer than remote.
        """
        if not self.service: self.authenticate()
        
        if not remote_name:
            remote_name = os.path.basename(local_path)
            
        # Check if exists
        query = f"name='{remote_name}' and '{folder_id}' in parents and trashed=false"
        results = self.service.files().list(q=query, spaces='drive', fields='files(id, modifiedTime)').execute()
        items = results.get('files', [])
        
        file_metadata = {
            'name': remote_name,
            'parents': [folder_id]
        }
        
        if items:
            file_id = items[0]['id']
            remote_time_str = items[0]['modifiedTime'] # RFC 3339
            
            if check_date:
                import dateutil.parser
                import datetime
                import pytz
                
                remote_mtime = dateutil.parser.parse(remote_time_str)
                local_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(local_path), pytz.UTC)
                
                # Add buffer of 2 seconds for filesystem diffs
                if local_mtime <= remote_mtime + datetime.timedelta(seconds=2):
                    print(f"Content {remote_name} is up to date on {self.account_email}. Skipping.")
                    return file_id

            if overwrite:
                print(f"Updating {remote_name} on {self.account_email}...")
                media = MediaFileUpload(local_path, resumable=True)
                updated_file = self.service.files().update(
                    fileId=file_id, 
                    media_body=media
                ).execute()
                return updated_file.get('id')
            else:
                 return file_id
        else:
            print(f"Uploading {remote_name} to {self.account_email}...")
            media = MediaFileUpload(local_path, resumable=True)
            file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            return file.get('id')

    def list_files(self, folder_id):
        """Lists files in a folder."""
        if not self.service: self.authenticate()
        
        query = f"'{folder_id}' in parents and trashed=false"
        results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        return results.get('files', [])

    def download_file(self, file_id, local_path, retries=3):
        """Downloads a file by ID with retry logic."""
        import time
        if not self.service: self.authenticate()
        
        for attempt in range(retries):
            try:
                request = self.service.files().get_media(fileId=file_id)
                fh = io.FileIO(local_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024*5) # 5MB chunks
                
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    # print(f"Download {int(status.progress() * 100)}%")
                
                # If we get here, download is complete
                return
                
            except Exception as e:
                print(f"Download error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1)) # Exponential backoff
                else:
                    raise e
            finally:
                if 'fh' in locals():
                    fh.close()

    def get_file_link(self, file_id):
        """Returns the webViewLink for a file."""
        if not self.service: self.authenticate()
        
        file = self.service.files().get(fileId=file_id, fields='webViewLink').execute()
        return file.get('webViewLink')
