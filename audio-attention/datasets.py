# dictionary of data details
database = {}

# standard datasets used for ACL2018 challenge
database['MOSEI_acl2018'] = {}
database['MOSEI_acl2018']['scp_covarep'] = '/share/spandh.ami1/emotion/tools/audioemotion/audio-attention/mosei.11866.covarep.scp'
database['MOSEI_acl2018']['fea_covarep'] = '/share/spandh.ami1/emotion//import/feat/converthdf5numpy/covarep_clipped_11866.npy'
database['MOSEI_acl2018']['scp_fbk'] = '/share/spandh.ami1/emotion/lib/flists/mosei/coding.fbk.10793.scp'
database['MOSEI_acl2018']['ref_npy'] = '/share/spandh.ami1/emotion/import/feat/converthdf5numpy/longest_clipped_emotion_labels_11866.npy'
database['MOSEI_acl2018']['ref_etm'] ='/share/spandh.ami1/emotion/lib/ref/mosei/make_ref/mosei.clipped0.5s.11866.etm'
database['MOSEI_acl2018']['ids'] = '/share/spandh.ami1/emotion/import/feat/converthdf5numpy/ids_11866.npy'

# Edinburgh' particpation in ACL2018 created different valid/test set split
database['MOSEI_edinacl2018'] = {}
database['MOSEI_edinacl2018']['scp_covarep'] = database['MOSEI_acl2018']['scp_covarep']
database['MOSEI_edinacl2018']['scp_fbk'] = database['MOSEI_acl2018']['scp_fbk']
database['MOSEI_edinacl2018']['ref_npy'] = database['MOSEI_acl2018']['ref_npy']
database['MOSEI_edinacl2018']['ref_etm'] = database['MOSEI_acl2018']['ref_etm']

# enterface, no official train/valid/test split
database['ent05p2'] = {}
database['ent05p2']['scp_fbk'] = '/share/spandh.ami1/emotion/lib/flists/enterface/enterface.complete.1287.scp'
database['ent05p2']['ref_etm'] = '/share/spandh.ami1/emotion/lib/ref/enterface/enterface.complete.1287.etm'

# enterface, no official train/valid/test split
database['misc'] = {}
database['misc']['scp'] = ''
database['misc']['ref'] = ''

