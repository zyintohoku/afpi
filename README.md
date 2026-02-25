synthesis image reconstruction:

`python skip_inv.py --delta_threshold 5e-13 --output 'filename'`


real image reconstruction:

`python skip_inv_real.py --output 'filename'`

prompt-to-prompt editing(requires "inv_latents.pt" file under 'filename'):

`python p2p.py --inv_method 'filename'`
