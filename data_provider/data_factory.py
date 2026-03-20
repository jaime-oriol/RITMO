"""
Factory de datos: crea datasets y dataloaders según configuración.
Patrón Factory para abstraer la creación de diferentes tipos de datasets.
"""

from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom  # Clases Dataset
from torch.utils.data import DataLoader  # DataLoader de PyTorch

# Diccionario que mapea nombres a clases de Dataset
data_dict = {
    'ETTh1': Dataset_ETT_hour,       # ETT horario
    'ETTh2': Dataset_ETT_hour,       # ETT horario variante
    'Weather': Dataset_Custom,        # Weather (10-min, split 70/10/20)
    'Electricity': Dataset_Custom,    # Consumo eléctrico MT_320
    'Traffic': Dataset_Custom,        # Ocupación sensores tráfico
    'Exchange': Dataset_Custom,       # Tipos de cambio
    'custom': Dataset_Custom,         # CSV genérico
}


def data_provider(args, flag):
    """
    Factory function que crea dataset y dataloader.

    Args:
        args: Configuración (data, batch_size, etc.)
        flag: 'train', 'val', o 'test'

    Returns:
        Tupla (dataset, dataloader)
    """
    Data = data_dict[args.data]  # Seleccionar clase
    timeenc = 0 if args.embed != 'timeF' else 1  # Tipo encoding temporal

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    # Crear dataset
    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    print(flag, len(data_set))

    # Crear DataLoader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
