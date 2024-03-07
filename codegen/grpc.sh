module="vector_service"
dest="pinecone/core/grpc/protos"
module_prefix="pinecone.core.grpc.protos"	

head -n 4 "${dest}/${module}_pb2_grpc.py" > "${dest}/tmp.py"
echo "import ${module_prefix}.${module}_pb2 as $(echo "${module}_pb2" | sed 's/_/__/g')" >> "${dest}/tmp.py"
tail -n +7 "${dest}/${module}_pb2_grpc.py" >> "${dest}/tmp.py"
mv "${dest}/tmp.py" "${dest}/${module}_pb2_grpc.py"