import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, TanhDiagGaussian, DiagGaussian, Actor

class Controller:
    def __init__(self, args):
        # register an actor
        #actor_backbone = MLP(input_dim=args.obs_dim, hidden_dims=args.hidden_dims)
        actor_backbone = MLP(input_dim=args.obs_dim, hidden_dims=args.hidden_dims, dropout_rate=getattr(args, 'dropout_rate', None))
        if not args.stochastic_actor:
            self.actor = Actor(actor_backbone, args.action_dim, max_action=args.max_action, device=args.device)
        else:
            # Determine which distribution to use based on args
            self.use_diag_gaussian = args.use_diag_gaussian

            if self.use_diag_gaussian:
                # For IQL: use DiagGaussian (unbounded=False, conditioned_sigma=False)
                dist = DiagGaussian(
                    latent_dim=getattr(actor_backbone, "output_dim"),
                    output_dim=args.action_dim,
                    unbounded=False,
                    conditioned_sigma=False,  # IQL uses fixed sigma_param
                    max_mu=args.max_action
                )
            else:
                # For other algorithms: use TanhDiagGaussian (conditioned_sigma=True)
                dist = TanhDiagGaussian(
                    latent_dim=getattr(actor_backbone, "output_dim"),
                    output_dim=args.action_dim,
                    unbounded=True,
                    conditioned_sigma=True,  
                    max_mu=args.max_action
                )
            self.actor = ActorProb(actor_backbone, dist, args.device)


        # load up weights
        cwd = os.getcwd()
        actor_path = os.path.join(cwd, args.actor_path, 'checkpoint', 'policy.pth')
        checkpoint = torch.load(actor_path, map_location=args.device)

        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("critic"):
                continue
            state_dict[k.replace("actor.", "")] = v

        #self.actor.load_state_dict(state_dict, strict=False)
        incompatible_keys = self.actor.load_state_dict(state_dict, strict=False)

        if incompatible_keys.missing_keys:
            print("\n--- Missing Keys---")
            print(incompatible_keys.missing_keys)

        if incompatible_keys.unexpected_keys:
            print("\n--- Unexpected Keys ---")
            print(incompatible_keys.unexpected_keys)

        #self.actor.load_state_dict(state_dict)
        self.actor.eval() # evaluation only

        # important arguments
        self.deterministic_mode = args.deterministic_mode
        self.stochastic_actor = args.stochastic_actor

    def act(self, obs):
        with torch.no_grad():
            if not self.stochastic_actor:
                action = self.actor(obs)
            else:
                dist = self.actor(obs)
                if self.deterministic_mode:
                    if self.use_diag_gaussian:
                        # DiagGaussian.mode() returns single value
                        action = dist.mode()
                    else:
                        # TanhDiagGaussian.mode() returns (action, raw_action)
                        action, _ = dist.mode()
                else:
                    if self.use_diag_gaussian:
                        # DiagGaussian doesn't have rsample, use sample
                        action = dist.sample()
                    else:
                        # TanhDiagGaussian.rsample() returns (action, raw_action)
                        action, _ = dist.rsample()

        return action.cpu().numpy()