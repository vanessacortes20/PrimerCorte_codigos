import sys
import os
import io
import contextlib

print("Configuring matplotlib for headless execution...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

original_show = plt.show

def new_show(*args, **kwargs):
    prefix = os.environ.get('SCRIPT_PREFIX', 'fig')
    out_dir = r"c:\Users\User\OneDrive\Escritorio\PrimerCorte_codigos\Redes_Neuronales\resultados"
    for i in plt.get_fignums():
        fig = plt.figure(i)
        fig.savefig(os.path.join(out_dir, f"{prefix}_fig_{i}.png"))
        print(f"Saved figure {i} to {prefix}_fig_{i}.png")
    plt.close('all')

plt.show = new_show

if __name__ == '__main__':
    script_path = sys.argv[1]
    file_name = os.path.basename(script_path)
    prefix = file_name.replace('.py', '')
    os.environ['SCRIPT_PREFIX'] = prefix
    
    out_dir = r"c:\Users\User\OneDrive\Escritorio\PrimerCorte_codigos\Redes_Neuronales\resultados"
    out_path = os.path.join(out_dir, f"{prefix}_salida.txt")
    
    print(f"Executing {file_name} and capturing output...")
    
    with open(out_path, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            print(f"--- Ejecución de {file_name} ---")
            print("="*50)
            
            with open(script_path, "r", encoding="utf-8") as s:
                code = s.read()
                
            try:
                # Add the folder itself to sys.path just in case
                script_dir = os.path.dirname(os.path.abspath(script_path))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                    
                namespace = {"__name__": "__main__", "__file__": script_path}
                exec(code, namespace)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            for i in plt.get_fignums():
                fig = plt.figure(i)
                fig.savefig(os.path.join(out_dir, f"{prefix}_fig_{i}_unshown.png"))
                print(f"Saved unshown figure {i} to {prefix}_fig_{i}_unshown.png")
            plt.close('all')
            
    print(f"Done executing {file_name}")
