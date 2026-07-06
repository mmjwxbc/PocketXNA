RNA_BASES = ['A', 'U', 'G', 'C']
DNA_BASES = ['A', 'T', 'G', 'C']

BASE_TO_ID_RNA = {base: i for i, base in enumerate(RNA_BASES)}
BASE_TO_ID_DNA = {base: i for i, base in enumerate(DNA_BASES)}

ATOM_ROLES = ['phosphate', 'sugar', 'base']
ATOM_ROLE_TO_ID = {role: i for i, role in enumerate(ATOM_ROLES)}

PHOSPHATE_ATOMS = {'P', 'OP1', 'OP2', 'O1P', 'O2P', "O5'"}

SUGAR_ATOMS = {
    "C1'", "C2'", "C3'", "C4'", "C5'",
    "O2'", "O3'", "O4'", "O5'",
}

BASE_ATOMS = {
    'N1', 'C2', 'N3', 'C4', 'C5', 'C6',
    'N7', 'C8', 'N9',
    'O2', 'O4', 'O6',
    'N2', 'N4', 'N6',
}

STANDARD_RNA_RESNAMES = {
    'A': 'A',
    'U': 'U',
    'G': 'G',
    'C': 'C',
    'RA': 'A',
    'RU': 'U',
    'RG': 'G',
    'RC': 'C',
}

STANDARD_DNA_RESNAMES = {
    'DA': 'A',
    'DT': 'T',
    'DG': 'G',
    'DC': 'C',
}
