import gzip
from model import *
from pdb_utils import *
from Bio.PDB.Polypeptide import index_to_one
import argparse
import warnings
from Bio import BiopythonDeprecationWarning
import os
import pandas as pd

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

script_dir = os.path.dirname(__file__)


def get_torch_model(ckpt_path, device="cpu"):
    model = AMPNN(
        embed_dim=128,
        edge_dim=27,
        node_dim=28,
        dropout=0.2,
        layer_nums=3,
        token_num=21,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    model.eval()
    model.to(device)
    return model


def cal_plddt(pdb_file):
    bs = []
    if pdb_file.endswith(".pdb.gz"):
        with gzip.open(pdb_file, "rt") as f:
            for line in f:
                if line.startswith("ATOM"):
                    b = float(line[60:66])
                    bs.append(b)
    if pdb_file.endswith(".pdb"):
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    b = float(line[60:66])
                    bs.append(b)
    return np.mean(bs)


def make_one_scan(
    pdb_file, torch_models: list, device="cpu", save_pt=False, save_dir=None
):
    if not save_dir:
        save_dir = os.path.dirname(pdb_file)
    protbb = read_pdb_to_protbb(pdb_file)
    node, edge, seq = get_neighbor(protbb, noise_level=0.0)
    probs = []
    with torch.no_grad():
        for torch_model in torch_models:
            # torch_model=torch_model.to(device)
            logits, _ = torch_model(node.to(device), edge.to(device))
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            probs.append(prob)
    if save_pt:
        data_dict = {}
        for pos, aa in enumerate(protbb.seq):
            energy = np.zeros(21)
            aa_index = int(aa.item())
            one_letter_aa = index_to_one(aa_index)
            for prob in probs:
                energy += -np.log(prob[pos] / prob[pos][aa_index])
            data_dict[f"{one_letter_aa}_{pos+1}"] = np.float16(energy)
        pt_path = os.path.join(
            save_dir, os.path.basename(pdb_file).replace(".pdb", "_pred_mask.pt")
        )
        torch.save(data_dict, pt_path)
        print(f"save {pt_path}")
    else:
        PSSM_Alphabet = "ARNDCQEGHILKMFPSTWYV"
        Bio_Alphabet = "".join([index_to_one(i) for i in range(20)])
        energy_data_list = []

        txt_path = os.path.join(
            save_dir, os.path.basename(pdb_file).replace(".pdb", "_pred_mask.txt")
        )
        with open(
            txt_path,
            "w",
        ) as f:
            for pos, aa in enumerate(protbb.seq):
                energy = np.zeros(21)
                for prob in probs:
                    energy += -np.log(prob[pos] / prob[pos][int(aa.item())])
                for i in range(20):
                    f.write(
                        f"{index_to_one(int(aa.item()))}{pos+1}{index_to_one(i)} {energy[i]}\n"
                    )

                # Convert energy values to a dictionary
                energy_dict = dict(zip(Bio_Alphabet, energy))

                energy_dict["pos"] = pos

                # Create a DataFrame from the dictionary
                energy_df = pd.DataFrame([energy_dict])

                # Reorder the DataFrame based on PSSM_Alphabet
                energy_df = energy_df.reindex(columns=list(PSSM_Alphabet) + ["pos"])

                # Append the DataFrame to the list
                energy_data_list.append(energy_df)

            # Concatenate the list of DataFrames into one DataFrame
            energy_data = pd.concat(energy_data_list, ignore_index=True)

            # Set the "pos" column as the row index
            energy_data = energy_data.set_index("pos")

            # Transpose the DataFrame and reset the index
            energy_data = energy_data.T.reset_index()

            # Save the DataFrame to a CSV file
            csv_path = os.path.join(
                save_dir,
                os.path.basename(pdb_file).replace(".pdb", "_pred_mask.csv"),
            )
            energy_data.to_csv(
                csv_path,
                index=False,
            )
            print(f"Saved {csv_path} , {txt_path}")


def main(args):
    pdb_filename = args.pdb_filename
    device = args.device
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)


    device = 'cuda' if torch.cuda.is_available() and device =='cuda' else 'cpu'
    device= torch.device(device)
    print(f'using device {device}')
    torch_model_c = get_torch_model(
        os.path.join(script_dir, "..", "pythia-c.pt"), device=device
    )
    torch_model_p = get_torch_model(
        os.path.join(script_dir, "..", "pythia-p.pt"), device=device
    )


    if pdb_filename:
        make_one_scan(
            pdb_file=pdb_filename,
            torch_models=[torch_model_c, torch_model_p],
            device=device,
            save_pt=False,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command line interface for the given code."
    )
    parser.add_argument(
        "--pdb_filename",
        type=str,
        default=None,
        help="Path to a specific PDB filename.",
    )
    
    parser.add_argument("--device", type=str, default="cpu", help="Try to use cpu")
    parser.add_argument(
        "--save_dir", type=str, default="./pythia_predictions", help="Set save dir."
    )

    args = parser.parse_args()
    main(args)
