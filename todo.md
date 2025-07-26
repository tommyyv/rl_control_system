# todo (get it good enough)
## in-progress

- [x] add system/global parameters => what are they?
- [x] sanitize & handle data
- [x] add system identifier => train model to simulate continous operation
- [x] add data loader
- [x] add pid controller => need to test
- [x] add model env => does this already include anomaly generators?
- [x] add rl algo
- [x] add run_baseline_evaluation()
- [x] add run_hybrid_evaluation()
- [x] add analyze_results()
- [x] add plot_results()
- [x] add anomaly generator
- [x] add anomaly detector
- [x] bug: fix stuff => lol.
- [x] refactor code to make it look sensible
- [x] document code files

## backlog
- [ ] add error handling
- [ ] unit test
- [ ] add types
- [ ] verify adherence to defensive & secure programming

## sop (seq. of ops)
### phase 1: setup & system modeling
1. define config (good enough)
2. initialize system & global parameters (good enough)
3. load data & system identifier (good enough)
### phase 2: experimetation & data collection
4. create training env using gymnasium (good enough)
5. run baseline test (pid controller) (good enough)
6. document baseline test; run for x trials (good enough)
7. apply rl algorithm using sb3; run for x trials (good enough)
8. document results (good enough)
### phase 3: documentation - analysis & reporting
9. evaluate & analysis results: baseline vs hybrid (good enough)
10. write research paper; do this concurrently with experiment (good enough)

## lessons learned.
- prime example of 'it worked on my machine' => began with streamlining a base
development env, such as docker or cloud vm and then build the application in it
