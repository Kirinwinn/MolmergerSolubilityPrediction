"""DGL-based AttentiveFP model used by the MolMerger notebook."""
import scipy.stats as st

if not hasattr(st, "gilbrat") and hasattr(st, "gibrat"):
    st.gilbrat = st.gibrat
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.losses import L2Loss, Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel
from dgl.nn.pytorch import edge_softmax


class AttentiveGRU1(nn.Module):
    """Update node features while incorporating edge information."""

    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super().__init__()
        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        self.edge_transform[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata["e"] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_e("e", "m"), fn.sum("m", "c"))
        context = F.elu(g.ndata["c"])
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    """Update node features during AttentiveFP message passing."""

    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super().__init__()
        self.project_node = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        self.project_node[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata["a"] = edge_softmax(g, edge_logits)
        g.ndata["hv"] = self.project_node(node_feats)
        g.update_all(fn.u_mul_e("hv", "a", "m"), fn.sum("m", "c"))
        context = F.elu(g.ndata["c"])
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    """Initial AttentiveFP message passing layer using edge features."""

    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super().__init__()
        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size), nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU(),
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * graph_feat_size, 1), nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(
            graph_feat_size, graph_feat_size, graph_feat_size, dropout
        )

    def reset_parameters(self):
        self.project_node[0].reset_parameters()
        self.project_edge1[0].reset_parameters()
        self.project_edge2[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges1(self, edges):
        return {"he1": torch.cat([edges.src["hv"], edges.data["he"]], dim=1)}

    def apply_edges2(self, edges):
        return {"he2": torch.cat([edges.dst["hv_new"], edges.data["he1"]], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata["hv"] = node_feats
        g.ndata["hv_new"] = self.project_node(node_feats)
        g.edata["he"] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata["he1"] = self.project_edge1(g.edata["he1"])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata["he2"])

        return self.attentive_gru(g, logits, g.edata["he1"], g.ndata["hv_new"])


class GNNLayer(nn.Module):
    """AttentiveFP message passing layer over node representations."""

    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super().__init__()
        self.project_edge = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * node_feat_size, 1), nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges(self, edges):
        return {"he": torch.cat([edges.dst["hv"], edges.src["hv"]], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata["hv"] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata["he"])
        return self.attentive_gru(g, logits, node_feats)


class AttentiveFPGNN(nn.Module):
    """AttentiveFP graph neural network."""

    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        num_layers=2,
        graph_feat_size=200,
        dropout=0.0,
    ):
        super().__init__()
        self.init_context = GetContext(
            node_feat_size, edge_feat_size, graph_feat_size, dropout
        )
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def reset_parameters(self):
        self.init_context.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        return node_feats


class GlobalPool(nn.Module):
    """One-step AttentiveFP readout."""

    def __init__(self, feat_size, dropout):
        super().__init__()
        self.compute_logits = nn.Sequential(
            nn.Linear(2 * feat_size, 1), nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat_size, feat_size)
        )
        self.gru = nn.GRUCell(feat_size, feat_size)

    def forward(self, g, node_feats, g_feats, get_node_weight=False):
        with g.local_scope():
            g.ndata["z"] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1)
            )
            g.ndata["a"] = dgl.softmax_nodes(g, "z")
            g.ndata["hv"] = self.project_nodes(node_feats)

            g_repr = dgl.sum_nodes(g, "hv", "a")
            context = F.elu(g_repr)

            if get_node_weight:
                return self.gru(context, g_feats), g.ndata["a"]
            return self.gru(context, g_feats)


class AttentiveFPReadout(nn.Module):
    """AttentiveFP readout from node features to graph features."""

    def __init__(self, feat_size, num_timesteps=2, dropout=0.0):
        super().__init__()
        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats, get_node_weight=False):
        with g.local_scope():
            g.ndata["hv"] = node_feats
            g_feats = dgl.sum_nodes(g, "hv")

        if get_node_weight:
            node_weights = []

        for readout in self.readouts:
            if get_node_weight:
                g_feats, node_weights_t = readout(
                    g, node_feats, g_feats, get_node_weight
                )
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats)

        if get_node_weight:
            return g_feats, node_weights
        return g_feats


class AttentiveFPPredictor(nn.Module):
    """AttentiveFP predictor for graph-level tasks."""

    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        num_layers=2,
        num_timesteps=2,
        graph_feat_size=200,
        n_tasks=1,
        dropout=0.0,
    ):
        super().__init__()
        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout,
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        self.predict = nn.Sequential(nn.Dropout(dropout), nn.Linear(graph_feat_size, n_tasks))

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        node_feats = self.gnn(g, node_feats, edge_feats)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats), node_weights
        g_feats = self.readout(g, node_feats, get_node_weight)
        return self.predict(g_feats)


class AttentiveFP(nn.Module):
    """DGL-based AttentiveFP for graph property prediction."""

    def __init__(
        self,
        n_tasks: int,
        num_layers: int = 2,
        num_timesteps: int = 2,
        graph_feat_size: int = 200,
        dropout: float = 0.0,
        mode: str = "regression",
        number_atom_features: int = 32,
        number_bond_features: int = 19,
        n_classes: int = 2,
        nfeat_name: str = "x",
        efeat_name: str = "edge_attr",
    ):
        try:
            import dgllife  # noqa: F401
        except ImportError as exc:
            raise ImportError("This class requires dgllife.") from exc

        if mode not in ["classification", "regression"]:
            raise ValueError("mode must be either 'classification' or 'regression'")

        super().__init__()
        self.n_tasks = n_tasks
        self.mode = mode
        self.n_classes = n_classes
        self.nfeat_name = nfeat_name
        self.efeat_name = efeat_name
        out_size = n_tasks * n_classes if mode == "classification" else n_tasks

        self.model = AttentiveFPPredictor(
            node_feat_size=number_atom_features,
            edge_feat_size=number_bond_features,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            graph_feat_size=graph_feat_size,
            n_tasks=out_size,
            dropout=dropout,
        )

    def forward(self, g):
        node_feats = g.ndata[self.nfeat_name]
        edge_feats = g.edata[self.efeat_name]
        out = self.model(g, node_feats, edge_feats)

        if self.mode == "classification":
            if self.n_tasks == 1:
                logits = out.view(-1, self.n_classes)
                softmax_dim = 1
            else:
                logits = out.view(-1, self.n_tasks, self.n_classes)
                softmax_dim = 2
            proba = F.softmax(logits, dim=softmax_dim)
            return proba, logits
        return out


class AttentiveFPModel(TorchModel):
    """DeepChem TorchModel wrapper for AttentiveFP."""

    def __init__(
        self,
        n_tasks: int,
        num_layers: int = 2,
        num_timesteps: int = 2,
        graph_feat_size: int = 200,
        dropout: float = 0.0,
        mode: str = "regression",
        number_atom_features: int = 32,
        number_bond_features: int = 19,
        n_classes: int = 2,
        self_loop: bool = True,
        **kwargs,
    ):
        model = AttentiveFP(
            n_tasks=n_tasks,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            graph_feat_size=graph_feat_size,
            dropout=dropout,
            mode=mode,
            number_atom_features=number_atom_features,
            number_bond_features=number_bond_features,
            n_classes=n_classes,
        )
        if mode == "regression":
            loss: Loss = L2Loss()
            output_types = ["prediction"]
        else:
            loss = SparseSoftmaxCrossEntropy()
            output_types = ["prediction", "loss"]
        super().__init__(model, loss=loss, output_types=output_types, **kwargs)
        self._self_loop = self_loop

    def _prepare_batch(self, batch):
        inputs, labels, weights = batch
        dgl_graphs = [
            graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
        ]
        inputs = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super()._prepare_batch(([], labels, weights))
        return inputs, labels, weights
