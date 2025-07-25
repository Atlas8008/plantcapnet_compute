import torch
from torch import nn

from functools import partial


class AsymmetricClassMapper(nn.Module):
    """Maps a list of classes to a subset of classes. This makes it possible to extract classifications for a subset of classes without changing the model architecture.
    """
    def __init__(self, full_list, sub_list):
        """
        Args:
            full_list (list): List of all classes.
            sub_list (list): List of classes to be extracted.
        """
        super().__init__()
        self.full_list = full_list
        self.sub_list = sub_list

        sub_list_indices = torch.tensor([self.full_list.index(it) for it in sub_list])

        self.register_buffer("sub_list_indices", sub_list_indices)

    def __call__(self, x):
        if not isinstance(x, (tuple, list)):
            x = [x]

        x = [torch.index_select(x_val, dim=1, index=self.sub_list_indices) for x_val in x]

        if len(x) == 1:
            x = x[0]

        return x


# Implemented example mappers
MAPPERS = {
    "iadiv2ia": partial(AsymmetricClassMapper, full_list=sorted([
        "Acer L.",
        "Achillea millefolium",
        "Agrimonia eupatoria",
        "Ajuga reptans L.",
        "Allium vineale",
        "Anthriscus sylvestris (L.) Hoffm.",
        "Arenaria serpyllifolia",
        "Betula pendula",
        "Campanula rapunculoides L.",
        "Capsella bursa-pastoris",
        "Carpinus betulus L.",
        "Centaurea jacea",
        "Cerastium holosteoides",
        "Cirsium arvense",
        "Crepis biennis",
        "Crepis capillaris",
        "Daucus carota",
        "Descurainia sophia",
        "Dianthus carthusianorum",
        "Draba verna",
        "Eryngium campestre",
        "Euonymus europaeus L.",
        "Falcaria vulgaris",
        "Filipendula vulgaris",
        "Fraxinus excelsior L.",
        "Galium album",
        "Galium mollugo L.",
        "Galium verum",
        "Geranium molle",
        "Geranium pratense",
        "Heracleum sphondylium L.",
        "Hieracium L.",
        "Knautia arvensis",
        "Lactuca serriola",
        "Lamium amplexicaule",
        "Lamium purpureum",
        "Lathyrus pratensis",
        "Leucanthemum vulgare",
        "Lotus corniculatus",
        "Lychnis",
        "Lysimachia nummularia L.",
        "Medicago falcata",
        "Medicago lupulina",
        "Myosotis ramosissima",
        "Picris hieracioides",
        "Pimpinella saxifraga",
        "Plantago lanceolata",
        "Plantago media",
        "Polygonatum multiflorum (L.) All.",
        "Prunella vulgaris",
        "Ranunculus acris",
        "Rumex acetosa L.",
        "Salvia pratensis",
        "Scabiosa canescens",
        "Scabiosa ochroleuca",
        "Scorzoneroides autumnalis",
        "Securigera varia",
        "Senecio vernalis",
        "Silene pratensis",
        "Smyrnium perfoliatum L.",
        "Sonchus asper",
        "Stachys recta",
        "Stellaria media",
        "Taraxacum officinale",
        "Taxus baccata L.",
        "Tilia cordata Mill.",
        "Tragopogon pratensis",
        "Trifolium dubium",
        "Trifolium pratense",
        "Trifolium repens",
        "Veronica arvensis",
        "Veronica chamaedrys",
        "Veronica hederifolia",
        "Veronica persica",
        "Veronica verna",
        "Vicia angustifolia L.",
        "Vicia hirsuta",
        "Viola arvensis",
        "Viola hirta",
        "Poaceae",
    ]),
    sub_list=[
        "Achillea millefolium",
        "Centaurea jacea",
        "Poaceae",
        "Lotus corniculatus",
        "Medicago lupulina",
        "Plantago lanceolata",
        "Scorzoneroides autumnalis",
        "Trifolium pratense",
    ]),
    "bgj21f2i": partial(AsymmetricClassMapper, full_list=[
        "Ace pse",
        "Aeg pod",
        "Aju rep",
        "Ane nem",
        "Ant syl",
        "Cam rap",
        "Cam rot",
        "Car bet",
        "Cer hol",
        "Crataegus sp.",
        "Cre bie",
        "Euo eur",
        "Fra exc",
        "Fra vir",
        "Gal mol",
        "Geu urb",
        "Hed hel",
        "Her sph",
        "Hieracium sp.",
        "Lat pra",
        "Lat ver",
        "Leu vul",
        "Lil mar",
        "Lis ova",
        "Lon xyl",
        "Lys num",
        "Med lup",
        "Pae per",
        "Pil off",
        "Pla lan",
        "Pla med",
        "Pol mul",
        "Pol odo",
        "Pri ver",
        "Pri vul",
        "Quercus sp.",
        "Ran acr",
        "Ran aur",
        "Ran bul",
        "Rosa sp.",
        "Rum ace",
        "Sco aut",
        "Smy per",
        "Tar off",
        "Tax bac",
        "Til cor",
        "Tri dub",
        "Tri pra",
        "Tri rep",
        "Ver cha",
        "Vic sat",
        "Vio hir",
        "grasses",
    ],
    sub_list=[
        "Ace pse",
        "Aeg pod",
        "Aju rep",
        "Ane nem",
        "Ant syl",
        "Cam rap",
        "Cam rot",
        "Cre bie",
        "Fra exc",
        "Fra vir",
        "Gal mol",
        "Geu urb",
        "Hieracium sp.",
        "Lat pra",
        "Lat ver",
        "Leu vul",
        "Lil mar",
        "Lis ova",
        "Pla med",
        "Pol mul",
        "Pri ver",
        "Pri vul",
        "Quercus sp.",
        "Ran acr",
        "Ran aur",
        "Ran bul",
        "Rum ace",
        "Smy per",
        "Tar off",
        "Tax bac",
        "Tri dub",
        "Tri pra",
        "Tri rep",
        "Ver cha",
        "Vic sat",
        "Vio hir",
        "grasses",
    ])
}

