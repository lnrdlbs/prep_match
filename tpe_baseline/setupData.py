import os
import pandas as pd
from sklearn.datasets import load_iris, load_wine

def download_datasets():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    datasets = {
        "iris": "builtin",
        "wine": "builtin",
        "glass": "https://www.openml.org/data/get_csv/41/glass.csv",
        "ecoli": "https://www.openml.org/data/get_csv/39/ecoli.csv",
        "ionosphere": "https://www.openml.org/data/get_csv/59/ionosphere.csv"
    }

    for name, url in datasets.items():
        file_path = os.path.join(data_dir, f"{name}.csv")

        if os.path.exists(file_path):
            print(f"[-] {name}.csv ist bereits vorhanden.")
            try:
                df = pd.read_csv(file_path)
                old_target = df.columns[-1]
                
                if old_target != "target":
                    df = df.rename(columns={old_target: "target"})
                    df.to_csv(file_path, index=False)
                    print(f"    -> Bereits existierende Daten aktualisiert: '{old_target}' wurde in 'target' umbenannt.")
                else:
                    print(f"    -> Bereits korrekt konfiguriert.")
            except Exception as e:
                print(f"    Fehler beim Aktualisieren von {name}: {e}")
            continue

        print(f"[+] Lade {name}...")
        try:
            if url == "builtin":
                if name == "iris":
                    data = load_iris(as_frame=True)
                elif name == "wine":
                    data = load_wine(as_frame=True)
                
                df = data.frame
            else:
                df = pd.read_csv(url)
            
            old_target = df.columns[-1]
            if old_target != "target":
                df = df.rename(columns={old_target: "target"})
            
            df.to_csv(file_path, index=False)
            print(f"    Erfolgreich geladen und mit Zielspalte 'target' gespeichert!")
            
        except Exception as e:
            print(f"    Fehler bei {name}: {e}")

    print("\nSlay")    

if __name__ == "__main__":
    download_datasets()