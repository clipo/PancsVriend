#!/bin/bash
foldername="llm-schelling" # remote folder

echo ""
echo "syncing ${foldername} up to spiedie"

rsync -vaurP . apape@spiedie.binghamton.edu:PM/${foldername}   # syncs "up"

sleep 1
echo ""
echo "syncing ${foldername} down from  to spiedie"

rsync -vaurP apape@spiedie.binghamton.edu:PM/${foldername}/ .  # syncs "down"
