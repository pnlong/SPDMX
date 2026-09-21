# MedleyDB metadata lives on deepfreeze, not in git

Default path: `/deepfreeze/share/pnlong/MedleyDB/Metadata`
(override with `SPDMX_MEDLEYDB_METADATA`).

YAML files come from [marl/medleydb](https://github.com/marl/medleydb)
(`medleydb/data/Metadata`). Refresh onto deepfreeze:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/marl/medleydb.git /tmp/medleydb_meta
cd /tmp/medleydb_meta && git sparse-checkout set medleydb/data/Metadata
rm -rf /deepfreeze/share/pnlong/MedleyDB/Metadata
cp -a /tmp/medleydb_meta/medleydb/data/Metadata /deepfreeze/share/pnlong/MedleyDB/Metadata
```
