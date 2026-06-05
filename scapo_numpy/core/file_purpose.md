## skinning.py

### Stage 1: Skinning Field

- Each point gets a probability of belonging to each bone:

        - W_ip = exp(-D**2/gamma)/sum(-D**2/gamma)
        where D_ip is the mahanlanobis distance.

        The mahalanobis distance is:

        D^2= (s_i - O_p)T. Q_p.(s_i - O_p)
        - This creates soft ownership instead of hard assigment.