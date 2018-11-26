# ditionary of data details
database = {}

database['MOSEI_acl2018'] = {}
database['MOSEI_acl2018']['scp_covarep'] = '/share/spandh.ami1/emotion/tools/audioemotion/audio-attention/mosei.11866.covarep.scp'
database['MOSEI_acl2018']['fea_covarep'] = '/share/spandh.ami1/emotion//import/feat/converthdf5numpy/covarep_clipped_11866.npy'
database['MOSEI_acl2018']['ref'] = '/share/spandh.ami1/emotion/import/feat/converthdf5numpy/longest_clipped_emotion_labels_11866.npy'
database['MOSEI_acl2018']['ids'] = '/share/spandh.ami1/emotion/import/feat/converthdf5numpy/ids_11866.npy'

database['MOSEI_edinacl2018'] = {}
database['MOSEI_edinacl2018']['scp_covarep'] = database['MOSEI_acl2018']['scp_covarep']
database['MOSEI_edinacl2018']['ref'] = database['MOSEI_acl2018']['ref']

database['ent05p2'] = {}
database['ent05p2']['scp'] = ''
database['ent05p2']['ref'] = ''

database['misc'] = {}
database['misc']['scp'] = ''
database['misc']['ref'] = ''

