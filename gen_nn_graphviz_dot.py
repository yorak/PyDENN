architecture = (4,8,5,3)
#architecture = (3,5,2)
classifier = True

connection_lines = []
prev_layer_nodes = []
print r"digraph G{"
print r"rankdir=LR;"
print r"ranksep=1.0;"
print r"splines=line;"
print r"node [shape=circle,style=filled,color=black,fontcolor=white];"

# virtual input
this_layer_nodes = []
for i in range(architecture[0]):
    node_str = "vi%d"%(i+1)
    print "%s [shape=point, style=invis];"%node_str
    this_layer_nodes.append(node_str)
print "  {rank=source "+" ".join(this_layer_nodes)+"};"
prev_layer_nodes = this_layer_nodes
    
#input
this_layer_nodes = []
print r"subgraph cluster_input {"
print "  style=invis;"
print "  label=\"input\";"
for i in range(architecture[0]):
    node_str = "i%d"%(i+1)
    connection_lines.append("%s->%s"%(prev_layer_nodes[i], node_str))
    print "  "+node_str+";"
    this_layer_nodes.append(node_str)
print "  {rank=same "+" ".join(this_layer_nodes)+"};"
print r"}"
prev_layer_nodes = this_layer_nodes
  
# hidden
symbols = "abcdefghjklmn"
for hidden_layer_i, hidden_size in enumerate(architecture[1:-1]):
    this_layer_nodes = []
    print r"subgraph cluster_hidden%d {"%hidden_layer_i
    print "  style=invis;"
    print "  label=\"hidden%d\";"%hidden_layer_i
    for i in range(hidden_size):
        node_str = "%s%d"%(symbols[hidden_layer_i],i+1)
        print "  "+node_str+";"
        for prev_l_node_str in prev_layer_nodes:
            connection_lines.append( prev_l_node_str+"->"+node_str+";")
        this_layer_nodes.append(node_str)
    print "  {rank=same "+" ".join(this_layer_nodes)+"};"
    print r"}"
    prev_layer_nodes = this_layer_nodes
    
#output
print r"subgraph cluster_output {"
print "  style=invis;"
print "  label=\"output\";"
this_layer_nodes = []
for i in range(architecture[-1]):
    node_str = "o%d"%(i+1)
    print "  "+node_str+";"
    for prev_l_node_str in prev_layer_nodes:
        connection_lines.append( prev_l_node_str+"->"+node_str+";")
    this_layer_nodes.append(node_str)
    print "  {rank=same "+" ".join(this_layer_nodes)+"};"
print r"}"
prev_layer_nodes = this_layer_nodes

    
if classifier:
    print('argmax [label="arg-\nmax",shape=rect,style=solid,fontcolor=black];')
    for out_node_str in prev_layer_nodes:
        connection_lines.append( out_node_str+"->argmax;")
    print("hidden [style=invis, shape=point];")
    print(r'argmax->hidden [label="clas-\nsifi-\ncation"];')

# connections
for connection in connection_lines:
    print connection

        



print r"}"
