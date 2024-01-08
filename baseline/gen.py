lenth = 24
for i in range(lenth):
    print(
        "# layer 0{}\n\
        action = policy[{}]\n\
        residual = x\n\
        if tf.reduce_sum(action) > 0.0:\n\
            action_mask = tf.reshape(action, (-1, 1, 1, 1))\n\
            fx = self.rnet.blocks[0][{}](x)\n\
            fx = tf.nn.relu(residual + fx)\n\
            x = fx * action_mask + residual * (1.0 - action_mask)\n\
        ".format(i,i,i)
    )