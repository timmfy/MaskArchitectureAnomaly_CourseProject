coco_to_cityscape = {

    # --- THINGS ---

    1: 11,   # person -> person
    2: 18,   # bicycle -> bicycle
    3: 13,   # car -> car
    4: 17,   # motorcycle -> motorcycle
    6: 15,   # bus -> bus
    7: 16,   # train -> train
    8: 14,   # truck -> truck
    10: 6,   # traffic light -> traffic light
    13: 7,   # stop sign -> traffic sign
 
    # --- ROAD & SIDEWALK ---
    149: 0,  # road -> road
    191: 1,  # pavement-merged -> sidewalk
    
    # --- BUILDING ---
    128: 2,  # house -> building
    197: 2,  # building-other-merged -> building
    
    # --- WALL ---
    171: 3,  # wall-brick -> wall
    175: 3,  # wall-stone -> wall
    176: 3,  # wall-tile -> wall
    177: 3,  # wall-wood -> wall
    199: 3,  # wall-other-merged -> wall
    
    # --- FENCE ---
    185: 4,  # fence-merged -> fence
    
    # --- VEGETATION ---
    184: 8,  # tree-merged -> vegetation
    193: 8,  # grass-merged -> vegetation
    
    # --- TERRAIN ---
    125: 9,  # gravel -> terrain
    154: 9,  # sand -> terrain
    158: 9,  # snow -> terrain
    192: 9,  # mountain-merged -> terrain
    194: 9,  # dirt-merged -> terrain
    198: 9,  # rock-merged -> terrain
    # --- SKY ---
    187: 10  # sky-other-merged -> sky

}