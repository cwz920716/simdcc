    for (int i = 0; i < 64; i += 1) {
        for (int j = 0; j < 96; j += 1) {
            for (int c2 = -2; c2 < 32; c2 += 1) {
                for (int c3 = 0; c3 < 32; c3 += 1) {
                    blurx(c2 + 1, c3) = in(c2 + 32*i + 1, c3 + 32*j) + in(c2 + 32*i + 1, c3 + 32*j - 1) + in(c2 + 32*i + 1, c3 + 32*j + 1);
                }
            }
            for (int c2 = 0; c2 < 32; c2 += 1) {
                for (int c3 = 0; c3 < 32; c3 += 1) {
                    out(c2 + 32*i, c3 + 32*j) = blurx(c2, c3) + blurx(c2 - 1, c3) + blurx(c2 + 1, c3);
                }
            }
        }
    }

