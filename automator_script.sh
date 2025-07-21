#!/bin/zsh
# Folder‑Action script – copy new recordings and erase them from the recorder

DEST="$HOME/Documents/Recordings"       # where you keep finished files

for mount_point in "$@"; do
    # Act only on the voice‑recorder volume
    [[ "$(basename "$mount_point")" == "IC RECORDER" ]] || continue

    SRC="$mount_point/REC_FILE/FOLDER01"
    [[ -d "$SRC" ]] || continue

    # Loop over every regular file in FOLDER01
    for file in "$SRC"/*; do
        [[ -f "$file" ]] || continue           # skip if not a plain file

        # Copy it; if (and only if) that succeeds, delete the original
        rsync -a --ignore-existing "$file" "$DEST"/ && rm "$file"
    done

    # OPTIONAL: tidy up empty sub‑directories after deletion
    find "$SRC" -type d -empty -delete
done
