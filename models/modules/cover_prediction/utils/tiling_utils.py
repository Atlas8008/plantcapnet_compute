import torch


def sample_tiles(img, tile_size, n_tiles):
    ts = tile_size
    counts = torch.zeros(
        (img.shape[0], img.shape[-2], img.shape[-1])
    ).to(img.get_device())

    tiles = []
    x_coords = []
    y_coords = []
    
    for idx, batch_item in enumerate(img):
        
        x_vals = torch.randint(0, img.shape[-1] - ts, (n_tiles,))
        y_vals = torch.randint(0, img.shape[-2] - ts, (n_tiles,))
        
        for x, y in zip(x_vals, y_vals):
            tiles.append(batch_item[:, y:y + ts, x:x + ts])
            counts[idx, y:y + ts, x:x + ts] += 1
            
        x_coords.append(x_vals)
        y_coords.append(y_vals)
            
    tiles = torch.stack(tiles, dim=0)
            
    return tiles, counts, x_coords, y_coords
    
def join_and_discount_tiled_predictions(
    preds, counts, x_coords, y_coords, tile_size, n_tiles):      
        
    ts = tile_size
    
    preds = torch.reshape(
        preds, 
        (-1, n_tiles, preds.shape[-1]),
    )    
    
    preds_total = []
    
    for pred, count, x_vals, y_vals in zip(
        preds, counts, x_coords, y_coords):
        
        discounted_tile_preds = torch.zeros_like(pred[0])
        
        for tile_preds, x, y in zip(pred, x_vals, y_vals):
            local_counts = count[y:y + ts, x:x + ts]
            
            discounted_tile_preds += torch.sum(
                tile_preds[None, None] / local_counts[:, :, None], 
                dim=(0, 1)
            )
        
        preds_total.append(
            discounted_tile_preds / torch.sum(count != 0)
        )

    return torch.stack(preds_total, dim=0)