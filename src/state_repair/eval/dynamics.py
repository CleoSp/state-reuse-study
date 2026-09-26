"""Additional refinement forked from a deployed carry policy's own state."""
from pathlib import Path
import torch
from state_repair.execution.datasets import collate, frames
from state_repair.execution.durable import digest_json
from state_repair.models.policy import FixedBudgetPolicy
from .jobs import EvaluationJob
from .mechanism import tensor_hash, impact
from .metrics import score, input_hash


class DynamicsJob(EvaluationJob):
    def __init__(self, job: dict, root: Path):
        super().__init__(job, root)
        self.schedule=[(i,0) for i in range(len(self.roots))]

    @torch.no_grad()
    def step(self,index,batch,k):
        job=self.job
        stream=frames([self.roots[batch]],4,job['stream_seed'])
        policy=FixedBudgetPolicy(self.solver,self.adapter,job['source_K'])
        calls=0
        for f, values in enumerate(stream):
            obs=collate(values,None if f==0 else stream[f-1])[0].to(job['device'])
            previous=policy(obs); calls+=previous.block_calls
        prior=previous.state.detach()
        initial_actions=previous.prediction.logits.argmax(-1)[0].cpu().tolist()
        prior_hash=tensor_hash(prior.a,prior.z,previous.prediction.logits)
        from .mechanism import pack_prior
        saved=pack_prior(prior.a,prior.z,previous.prediction.logits)
        fraction=sum(impact(stream[-2][0],stream[-1][0]))/obs.valid_nodes.shape[1]
        records=[]
        for budget in job['budgets']:
            result=self.solver(obs,budget,prior.clone())
            calls+=result.block_calls
            actions=result.prediction.logits.argmax(-1)[0].cpu().tolist()
            records.append({'suite':job['suite'],'seed':job['seed'],'root_id':self.roots[batch].root_id,
                'frame':4,'branch':'carried-frame4','policy':'carry_refinement','K':budget,'source_K':job['source_K'],
                'source_policy':'carry','prefix_edits':4,'split':job['split'],'synthetic':job['synthetic'],
                'record_kind':'frozen_state_intervention','privileged':False,'deployable_policy':False,
                'checkpoint_sha256':self.checkpoint_hash,'checkpoint_job':self.source,'dataset_sha256':self.dataset_hash,
                'config_sha256':digest_json(job),'prior_state_sha256':prior_hash,'input_sha256':input_hash(stream[-1][0]),
                'actions':actions,'prior_actions':initial_actions,'prediction_sha256':digest_json(actions),
                'changed_action_fraction':sum(a!=b for a,b in zip(actions,initial_actions))/len(actions),
                'stratum':'low' if fraction<=.1 else 'high' if fraction>=.4 else 'middle','impact_fraction':fraction,
                'state_distance_from_prior':{'a':float((result.state.a-prior.a).norm()),'z':float((result.state.z-prior.z).norm())},
                'block_calls':result.block_calls,**score(stream[-1][0],actions)})
        return {'loss':0.,'forward_calls':calls,'records':records,'saved_prior':saved,'prior_state_sha256':prior_hash,
                'root_id':self.roots[batch].root_id,'branch':'carried-frame4'}
