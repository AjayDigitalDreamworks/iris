# Reference image credits

These reference images were downloaded from **Wikimedia Commons** so that the
snack/drink SKUs in the catalogue are *recognisable* out of the box. They are
**generic, openly-licensed sample photos**, not the operator's own products and
not photos of any specific shelf in this store. Replace them with a photo of the
actual product (Inventory → *Add image*, or
`scripts/add_product_images.py`) before relying on recognition in production.

Recognition quality is therefore only demonstrated for these sample images. IRIS
makes no accuracy claim against real in-store packaging until real reference
photos are supplied.

| File | Commons file | Author | Licence | Commons page |
| --- | --- | --- | --- | --- |
| `SNACK-CHIPS-01.jpg` | Tayto cheese and onion crisps | Samuel Wiki | CC0 1.0 | https://commons.wikimedia.org/wiki/File:Tayto_cheese_and_onion_crisps.jpg |
| `SNACK-NAMKEEN-01.jpg` | Spicy and namkeen garlic mixture snack with peanuts (West Bengal, India) | Billjones94 | CC0 1.0 | https://commons.wikimedia.org/wiki/File:Spicy_and_namkeen_garlic_mixture_snack_with_peanuts,_photographed_in_West_Bengal,_India,_August_13,_2024.jpg |
| `SNACK-BISCUIT-01.jpg` | Endless packets of chocolate biscuits | Leon Brooks | Public domain | https://commons.wikimedia.org/wiki/File:Endless_packets_of_chocolate_biscuits.jpg |
| `SNACK-CHOCO-01.jpg` | Rakhat Kazakhstan dark chocolate bar 01 | Duman Amangali, Central Asian Student Union (UBC) | CC BY 4.0 | https://commons.wikimedia.org/wiki/File:Rakhat_Kazakhstan_dark_chocolate_bar_01.jpg |
| `DRINK-COLA-01.jpg` | Coca-Cola glass bottle (Germany) | Ralf Roletschek (RalfR) | Public domain | https://commons.wikimedia.org/wiki/File:15-09-26-RalfR-WLC-0098_-_Coca-Cola_glass_bottle_(Germany).jpg |

`DEMO-APPLE-0{1,2,3}.jpg` are the pre-existing demo apple references and are not
covered by the table above.

## Local processing applied

Each download was orientation-corrected from its EXIF tag, converted to RGB, and
downscaled so the long edge is at most 1024 px (JPEG q92). Pixel content is
otherwise unmodified. The `incoming/` copies are the originals of the same
photos and are safe to delete.
