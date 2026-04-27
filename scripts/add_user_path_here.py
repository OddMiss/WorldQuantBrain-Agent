import os
import winreg
import ctypes

def add_current_dir_to_user_path():
    # 1. Get the exact directory where this python script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    key_path = r"Environment"
    
    try:
        # 2. Open the registry key
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_ALL_ACCESS)
        
        # 3. Get the current PATH
        try:
            current_path, _ = winreg.QueryValueEx(key, "PATH")
        except FileNotFoundError:
            current_path = ""

        # 4. Check if the directory is already in the PATH
        if current_dir not in current_path.split(os.pathsep):
            # Append it
            updated_path = current_path + os.pathsep + current_dir if current_path else current_dir
            winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, updated_path)
            
            print(f"[SUCCESS] Added to User PATH: \n{current_dir}\n")
            
            # 5. Broadcast the change to Windows
            print("[INFO] Refreshing Windows environment variables...")
            HWND_BROADCAST = 0xFFFF
            WM_SETTINGCHANGE = 0x001A
            ctypes.windll.user32.SendMessageTimeoutW(
                HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment", 0x0002, 5000, None
            )
            print("[SUCCESS] Refresh complete.")
            
        else:
            print(f"[NOTICE] This folder is already in your User PATH: \n{current_dir}\n")
            
    except Exception as e:
        print(f"[ERROR] Failed to update registry: {e}")
    finally:
        winreg.CloseKey(key)

if __name__ == "__main__":
    print("--- User PATH Updater ---")
    add_current_dir_to_user_path()