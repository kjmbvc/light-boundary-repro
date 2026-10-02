from pathlib import Path
import io,json,subprocess,sys,unittest
import test_model

BASE=Path(__file__).resolve().parent
stream=io.StringIO()
suite=unittest.defaultTestLoader.loadTestsFromModule(test_model)
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
subprocess.run([sys.executable,str(BASE/'geometry.py')],check=True)
geometry=json.loads((BASE/'geometry-results.json').read_text(encoding='utf-8'))
record={'physics_tests_run':result.testsRun,'physics_failures':len(result.failures),'physics_errors':len(result.errors),
        'physics_passed':result.wasSuccessful(),'synthetic_angle_cases_passed':geometry['passed'],
        'scope':'Physics tests check model implementation; angle tests do not calculate lux.',
        'physics_test_names':sorted(n for n in dir(test_model.PhysicsTests) if n.startswith('test_'))}
(BASE/'test-results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(stream.getvalue());print(json.dumps(record))
if not result.wasSuccessful(): raise SystemExit(1)
