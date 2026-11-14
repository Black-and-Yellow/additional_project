import time
from .entropy import cluster_answers
from .retrieval import BM25, extract

ABSTENTION = 'I could not find sufficient supporting evidence in the local corpus.'


class Pipeline:
    def __init__(self, store, model=None):
        self.store, self.model = store, model

    def run(self, query, mode='lookup', samples=5, threshold=0.65, seed=42, policy='always_retrieve', correction='guarded'):
        started = time.perf_counter()
        result = dict(query=query, mode=mode, policy=policy, correction=correction, threshold_nats=threshold, seed=seed,
                      implementation_version='1.1.0',
                      initial_answer=None, samples=[], uncertainty=None, diagnosis=None,
                      documents=[], answer=None, route=None, trace=[], warnings=[])
        def stage(name, detail):
            result['trace'].append({'stage': name, 'detail': detail, 'elapsed_ms': round((time.perf_counter() - started) * 1000, 2)})
        index = BM25(self.store.documents())
        if mode == 'lookup':
            docs = index.search(query)
            result.update(documents=docs, answer=extract(query, docs) or ABSTENTION,
                          route='extractive' if docs else 'abstain', supported=bool(docs))
            stage('retrieval', f'{len(docs)} relevant chunks from the local corpus')
            stage('answer', 'Verbatim passage selection' if docs else 'No relevant evidence')
            result['warnings'] = ['Document lookup uses BM25 and passage extraction. It does not run an LLM, semantic entropy, or internal probes. Relevance does not establish truth.']
        else:
            if self.model is None:
                raise RuntimeError('Local model is unavailable. Start with --model after downloading the model assets.')
            model = self.model
            calls, token_count = model.calls, model.generated_tokens
            result['model'] = model.model_id
            initial = model.generate(query, seed=seed)[0] if policy != 'always_retrieve' else None
            result['initial_answer'] = initial
            if initial is not None:
                stage('generation', 'Greedy answer without retrieved context')
            if policy == 'no_retrieval':
                result.update(answer=initial, route='accept_unverified', supported=False,
                    generation_calls=model.calls-calls, generated_tokens=model.generated_tokens-token_count,
                    latency_ms=round((time.perf_counter()-started)*1000,2))
                result['warnings'].append('Baseline generation without retrieval or evidence verification.')
                return self.store.save_run(result)
            uncertainty = None
            agrees = False
            if policy == 'adaptive':
                sampled = model.generate(query, count=samples, sample=True, seed=seed)
                uncertainty = cluster_answers(sampled, model.equivalent)
                uncertainty['method'] = 'Bidirectional NLI, complete-link clusters, empirical frequencies'
                result.update(samples=sampled, uncertainty=uncertainty)
                stage('uncertainty', f"{len(uncertainty['clusters'])} semantic clusters")
                # Check the greedy answer without including it in the entropy estimate.
                dominant = max(uncertainty['clusters'], key=lambda c: len(c['members']))
                agrees = model.equivalent(initial, dominant['label'])
            if policy == 'adaptive' and uncertainty['entropy_nats'] < threshold and agrees:
                result.update(answer=initial, route='accept_unverified', supported=False)
                result['warnings'].append('Low entropy means sample consistency, not factual correctness. This path has no evidence verification.')
                stage('gate', 'Accepted by the uncertainty rule without retrieval')
            else:
                docs = index.search(query)
                result['documents'] = docs
                stage('retrieval', f'{len(docs)} relevant chunks retrieved before context diagnosis')
                if not docs:
                    result.update(answer=ABSTENTION, route='abstain', supported=False)
                else:
                    candidate = model.generate(query, docs, seed=seed)[0]
                    result['context_answer'] = candidate
                    diagnosis = model.diagnose(query, docs, candidate) if correction != 'off' else None
                    result['diagnosis'] = diagnosis
                    if diagnosis:
                        stage('diagnosis', 'Measured attention/hidden-state ECS and residual-stream PKS')
                    original = candidate
                    original_supported = model.supports(original, docs)
                    flagged = diagnosis is not None and diagnosis.get('flagged', diagnosis['ecs'] < 0.5 and diagnosis['pks_nats'] > 0.05)
                    intervene = correction == 'guarded' and flagged and not original_supported
                    if intervene:
                        with model.intervention():
                            candidate = model.generate(query, docs, seed=seed)[0]
                        revised_supported = model.supports(candidate, docs)
                        result['intervention'] = {'alpha': 1.3, 'beta': 0.7, 'applied': True, 'accepted': revised_supported, 'candidate': candidate}
                        if not revised_supported:
                            candidate = original
                        stage('intervention', 'Scaled selected attention head and FFN outputs')
                    supported = original_supported if candidate == original else revised_supported
                    result.update(answer=candidate if supported else ABSTENTION, supported=supported,
                                  route=('aarf_experimental' if intervene and result['intervention']['accepted'] else 'retrieval') if supported else 'abstain',
                                  candidate_answer=candidate)
                    stage('verification', 'NLI evidence support passed' if supported else 'Support check failed, abstained')
                    result['warnings'].append('NLI support is an imperfect check. ' + ('The detector uses fitted context-consistency calibration.' if diagnosis and diagnosis.get('calibrated') else 'No calibrated detector is active.'))
            result['generation_calls'] = model.calls - calls
            result['generated_tokens'] = model.generated_tokens - token_count
        result['latency_ms'] = round((time.perf_counter() - started) * 1000, 2)
        return self.store.save_run(result)

