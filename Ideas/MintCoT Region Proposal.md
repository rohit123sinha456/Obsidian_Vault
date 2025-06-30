AS vineeth sasid, in mint cot paper the images are broken down into grids ( patches ) then thses patches are used with the trainable interleave token to get only the relavant patches for the reasinignn steps.

Sir said that for natural images breaking the image into such grid may not be such a good ideas. SO I propose why dont we use some region based proposal method, and use these proposed region with the interleaved token to get the representation of the region to be used in the reasoning process as defined in the mint cot.

Also since the texts are representation. interleave tokens are representation, images are representation why do we place the texts token then when an interleave token is generated we take the representation of the images. why cant we take all the tokens and train a positional encoding method which will learn how to place the tokens for bettwe reasoning in the intermediate steps

RegionCLIP: Region-based Language-Image Pretraining
Patch-based Selection and Refinement for Early Object Detection