# Add reference images for products (chips, snacks, anything)

Recognition works by comparing a detected crop against a product's **reference
image**. A product can only be named once it has one. There are two ways to add
them.

## 1. From the dashboard

Open **Inventory** → click a product → **Add reference image** and upload a
photo. The image is stored and its embedding computed immediately; no restart.

Or call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/products/<PRODUCT_ID>/reference-image \
     -F "file=@chips.jpg"
```

## 2. In bulk from a folder (this folder)

Name each photo after the product's SKU and run the importer:

```
data/product_catalog/incoming/
    SNACK-CHIPS-01.jpg
    SNACK-NAMKEEN-01.jpg
    SNACK-BISCUIT-01.jpg
```

```bash
cd /c/iris
backend/.venv/Scripts/python.exe scripts/add_product_images.py data/product_catalog/incoming --create
```

`--create` makes a product row for any SKU that does not exist yet, so a folder
of photos is enough to build the catalog. Then restart the backend (or call
`POST /api/v1/products/embeddings/rebuild`) so the running server reloads the
index.

## Photo tips

- One product per photo, filling most of the frame.
- Plain background, even lighting, no hand or shelf clutter.
- The packet front (the side a shopper sees), not the barcode side.
- Square-ish crops work best; keep files under a couple of MB.

## What this does and does not do

- **Does**: makes a product *recognisable* — it puts a NAME on a crop.
- **Does not**: make it *detectable*. The detector must still produce a box
  around the item. The COCO baseline is weak on packaged goods (a chips packet
  may be boxed as `book` or not at all), so for reliable boxes supply a
  fine-tuned retail model:

  ```bash
  # put the weights in models/ then:
  IRIS_DETECTOR_WEIGHTS=retail_best.pt
  ```

  Since Phase 10 a custom model's own class names (`chips`, `namkeen`, …) are
  treated as products automatically, so a fine-tune works end to end.
