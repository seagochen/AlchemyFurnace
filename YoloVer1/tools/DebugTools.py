def print_yolo_grids(frag_conf, frag_bbox, frag_classes):

    # shape of fragment
    B, C, N = frag_conf.shape

    # print out label of fragments
    for b in range(B):
        for n in range(N):
            print(frag_conf[b, :, n], frag_bbox[b, :, n], frag_classes[b, :, n])
        print("\n")
    print("===============================")


def print_fragments(batch_size=1, grids_size=1, *fragments):

    # print out label of fragments
    for b in range(batch_size):
        for n in range(grids_size):
            for frag in fragments:
                
                # print out fragments
                if len(frag.size()) == 3:
                    print(frag[b, :, n], end=" ")
                elif len(frag.size()) == 2:
                    print(frag[b, n], end=" ")

            print("\n")
    print("===============================")
